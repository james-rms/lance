// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Carry a serialized HNSW graph across a row-address remap.
//!
//! HNSW edges name local vector ids (row numbers in the partition storage),
//! not row addresses. Compaction rewrites row addresses and may delete rows,
//! but every storage `remap` keeps the surviving vectors in their original
//! order. So a remap never invalidates the surviving edges: without deletions
//! the graph is unchanged, and with deletions each surviving node only has to
//! be renumbered and lose its edges to deleted nodes.

use std::sync::Arc;

use arrow::array::{ArrayBuilder, AsArray, Float32Builder, ListBuilder, UInt32Builder};
use arrow::datatypes::{Float32Type, UInt32Type};
use arrow_array::{Array, RecordBatch};
use itertools::Itertools;
use lance_core::{Error, Result};

use super::builder::HNSW_METADATA_KEY;
use super::{HNSW, HnswMetadata, VECTOR_ID_COL};
use crate::vector::DIST_COL;
use crate::vector::graph::NEIGHBORS_COL;
use crate::vector::v3::subindex::IvfSubIndex;

/// Rewrite a serialized HNSW graph after some of its nodes were removed.
///
/// `graph` is the batch written by [`HNSW::to_batch`], including the distance
/// column. `new_ids[old_id]` is the id node `old_id` takes after the removal,
/// or `None` if it was removed; surviving ids must keep their relative order,
/// which is what every vector storage `remap` produces.
///
/// Surviving adjacency lists and their distances are kept, minus the edges to
/// removed nodes; nothing is re-linked, so node degree drops by the number of
/// removed neighbors. If the entry point is removed, the first surviving node
/// on the highest non-empty level replaces it, so search still starts from
/// the top of the graph.
///
/// `new_ids` must have one entry per graph node. If no node is removed, the
/// input batch is returned as is. Returns an empty batch if no node survives.
pub fn remap_graph_batch(graph: &RecordBatch, new_ids: &[Option<u32>]) -> Result<RecordBatch> {
    let metadata_json = graph
        .schema_ref()
        .metadata()
        .get(HNSW_METADATA_KEY)
        .ok_or_else(|| Error::index(format!("{HNSW_METADATA_KEY} not found in HNSW batch")))?;
    let metadata: HnswMetadata = serde_json::from_str(metadata_json).map_err(|e| {
        Error::index(format!(
            "Failed to decode HNSW metadata: {e}, json: {metadata_json}"
        ))
    })?;
    let num_nodes = match metadata.level_offsets.as_slice() {
        [start, end, ..] => end.saturating_sub(*start),
        _ => 0,
    };
    if num_nodes != new_ids.len() {
        return Err(Error::invalid_input(format!(
            "HNSW graph has {num_nodes} nodes but the remap covers {} nodes",
            new_ids.len()
        )));
    }
    let column = |name: &str| {
        graph.column_by_name(name).ok_or_else(|| {
            Error::index(format!(
                "HNSW batch has no {name} column; remapping needs every column the writer emits"
            ))
        })
    };
    let ids = column(VECTOR_ID_COL)?
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{VECTOR_ID_COL} must be UInt32")))?;
    let neighbors = column(NEIGHBORS_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?;
    let distances = column(DIST_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?;
    let neighbor_values = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?
        .values();
    let distance_values = distances
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?
        .values();
    if new_ids
        .iter()
        .enumerate()
        .all(|(old_id, new_id)| *new_id == Some(old_id as u32))
    {
        return Ok(graph.clone());
    }
    let new_id = |old_id: u32| new_ids.get(old_id as usize).copied().flatten();

    let num_rows = graph.num_rows();
    let mut id_builder = UInt32Builder::with_capacity(num_rows);
    let mut neighbors_builder = ListBuilder::with_capacity(
        UInt32Builder::with_capacity(neighbor_values.len()),
        num_rows,
    );
    let mut distances_builder = ListBuilder::with_capacity(
        Float32Builder::with_capacity(neighbor_values.len()),
        num_rows,
    );
    let mut level_offsets = Vec::with_capacity(metadata.level_offsets.len());
    level_offsets.push(0);
    // First surviving node of each level, to replace a removed entry point.
    let mut level_first_node = Vec::with_capacity(metadata.level_offsets.len());
    for (&start, &end) in metadata.level_offsets.iter().tuple_windows() {
        if start > end || end > num_rows {
            return Err(Error::index(format!(
                "HNSW level range {start}..{end} is invalid for a batch of {num_rows} rows"
            )));
        }
        let mut first_node = None;
        for row in start..end {
            let Some(node) = new_id(ids.value(row)) else {
                continue;
            };
            first_node.get_or_insert(node);
            id_builder.append_value(node);
            if !neighbors.is_null(row) {
                let edges = neighbors.value_offsets()[row] as usize
                    ..neighbors.value_offsets()[row + 1] as usize;
                let dists = distances.value_offsets()[row] as usize
                    ..distances.value_offsets()[row + 1] as usize;
                if edges.len() != dists.len() {
                    return Err(Error::index(format!(
                        "HNSW row {row} has {} neighbors but {} distances",
                        edges.len(),
                        dists.len()
                    )));
                }
                for (&neighbor, &dist) in neighbor_values[edges].iter().zip(&distance_values[dists])
                {
                    if let Some(neighbor) = new_id(neighbor) {
                        neighbors_builder.values().append_value(neighbor);
                        distances_builder.values().append_value(dist);
                    }
                }
            }
            neighbors_builder.append(true);
            distances_builder.append(true);
        }
        level_offsets.push(id_builder.len());
        level_first_node.push(first_node);
    }

    if id_builder.is_empty() {
        return Ok(RecordBatch::new_empty(HNSW::schema()));
    }
    let entry_point = match new_id(metadata.entry_point) {
        Some(entry_point) => entry_point,
        None => level_first_node
            .iter()
            .rev()
            .find_map(|node| *node)
            .ok_or_else(|| Error::internal("HNSW remap kept nodes on no level".to_string()))?,
    };
    let metadata = HnswMetadata {
        entry_point,
        params: metadata.params,
        level_offsets,
    };

    let mut schema_metadata = graph.schema_ref().metadata().clone();
    schema_metadata.insert(
        HNSW_METADATA_KEY.to_string(),
        serde_json::to_string(&metadata)?,
    );
    let schema = HNSW::schema()
        .as_ref()
        .clone()
        .with_metadata(schema_metadata);
    Ok(RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(neighbors_builder.finish()),
            Arc::new(distances_builder.finish()),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::array::AsArray;
    use arrow::compute::take;
    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array};
    use itertools::Itertools;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::Error;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use super::remap_graph_batch;
    use crate::vector::DIST_COL;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::graph::NEIGHBORS_COL;
    use crate::vector::hnsw::builder::{HNSW_METADATA_KEY, HnswBuildParams, HnswQueryParams};
    use crate::vector::hnsw::{HNSW, HnswMetadata, VECTOR_ID_COL};
    use crate::vector::v3::subindex::IvfSubIndex;

    const DIM: usize = 16;
    const TOTAL: usize = 2000;
    const K: usize = 10;

    fn build(total: usize) -> (FixedSizeListArray, HNSW) {
        let data = generate_random_array_with_seed::<Float32Type>(total * DIM, [7; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(data, DIM as i32).unwrap();
        let store = FlatFloatStorage::new(fsl.clone(), DistanceType::L2);
        let hnsw = HNSW::index_vectors(
            &store,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
        )
        .unwrap();
        (fsl, hnsw)
    }

    fn metadata(batch: &RecordBatch) -> HnswMetadata {
        serde_json::from_str(&batch.schema_ref().metadata()[HNSW_METADATA_KEY]).unwrap()
    }

    fn num_level_members(hnsw: &HNSW) -> usize {
        (0..hnsw.max_level() as usize)
            .map(|level| hnsw.num_nodes(level))
            .sum()
    }

    fn query_params() -> HnswQueryParams {
        HnswQueryParams {
            ef: 64,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        }
    }

    fn recall(hnsw: &HNSW, vectors: &FixedSizeListArray, queries: &FixedSizeListArray) -> f32 {
        let store = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
        let mut hits = 0;
        for i in 0..queries.len() {
            let query = queries.value(i);
            let truth = (0..vectors.len())
                .map(|j| {
                    let v = vectors.value(j);
                    let v = v.as_primitive::<Float32Type>();
                    let q = query.as_primitive::<Float32Type>();
                    let dist: f32 = v
                        .values()
                        .iter()
                        .zip(q.values())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (dist, j as u32)
                })
                .sorted_by(|a, b| a.0.total_cmp(&b.0))
                .take(K)
                .map(|(_, id)| id)
                .collect::<HashSet<_>>();
            let found = hnsw
                .search_basic(query, K, &query_params(), None, &store)
                .unwrap();
            hits += found.iter().filter(|n| truth.contains(&n.id)).count();
        }
        hits as f32 / (queries.len() * K) as f32
    }

    #[test]
    fn test_remap_without_removals_keeps_graph() {
        let (_, hnsw) = build(TOTAL);
        let batch = HNSW::load(hnsw.to_batch().unwrap())
            .unwrap()
            .to_batch()
            .unwrap();
        assert_eq!(batch.num_rows(), num_level_members(&hnsw));

        let new_ids = (0..TOTAL as u32).map(Some).collect::<Vec<_>>();
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        assert_eq!(remapped, batch);
    }

    #[rstest]
    #[case::one_percent(100)]
    #[case::ten_percent(10)]
    #[case::half(2)]
    fn test_remap_drops_removed_nodes(#[case] remove_every: usize) {
        let (vectors, hnsw) = build(TOTAL);
        let batch = hnsw.to_batch().unwrap();
        let old_metadata = metadata(&batch);

        let mut new_ids = Vec::with_capacity(TOTAL);
        let mut kept = Vec::new();
        for old_id in 0..TOTAL {
            if old_id % remove_every == 0 {
                new_ids.push(None);
            } else {
                new_ids.push(Some(kept.len() as u32));
                kept.push(old_id as u32);
            }
        }
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        let new_metadata = metadata(&remapped);
        assert_eq!(
            new_metadata.level_offsets.len(),
            old_metadata.level_offsets.len()
        );
        assert_eq!(new_metadata.level_offsets[1], kept.len());

        // Every level keeps exactly its surviving members, renumbered in order.
        let old_ids = batch[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let new_ids_col = remapped[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let old_levels = old_metadata.level_offsets.iter().tuple_windows::<(_, _)>();
        let new_levels = new_metadata.level_offsets.iter().tuple_windows::<(_, _)>();
        for (level, ((old_start, old_end), (new_start, new_end))) in
            old_levels.zip(new_levels).enumerate()
        {
            let expected = (*old_start..*old_end)
                .filter_map(|row| new_ids[old_ids.value(row) as usize])
                .collect::<Vec<_>>();
            let actual = new_ids_col.values()[*new_start..*new_end].to_vec();
            assert_eq!(actual, expected, "level {level}");
        }

        // No edge names a removed or out-of-range node, and each surviving
        // edge is an old edge with its distance carried along.
        let old_neighbors = batch[NEIGHBORS_COL].as_list::<i32>();
        let old_dists = batch[DIST_COL].as_list::<i32>();
        let new_neighbors = remapped[NEIGHBORS_COL].as_list::<i32>();
        let new_dists = remapped[DIST_COL].as_list::<i32>();
        let mut old_row = 0;
        for new_row in 0..remapped.num_rows() {
            while new_ids[old_ids.value(old_row) as usize].is_none() {
                old_row += 1;
            }
            let expected = old_neighbors
                .value(old_row)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .zip(
                    old_dists
                        .value(old_row)
                        .as_primitive::<Float32Type>()
                        .values(),
                )
                .filter_map(|(n, d)| new_ids[*n as usize].map(|n| (n, *d)))
                .collect::<Vec<_>>();
            let actual = new_neighbors
                .value(new_row)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied()
                .zip(
                    new_dists
                        .value(new_row)
                        .as_primitive::<Float32Type>()
                        .values()
                        .iter()
                        .copied(),
                )
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "row {new_row}");
            assert!(actual.iter().all(|(n, _)| (*n as usize) < kept.len()));
            old_row += 1;
        }

        let loaded = HNSW::load(remapped.clone()).unwrap();
        assert_eq!(loaded.len(), kept.len());
        assert_eq!(loaded.to_batch().unwrap(), remapped);
        assert_eq!(remapped.num_rows(), num_level_members(&loaded));

        let kept_vectors = take(&vectors, &UInt32Array::from(kept), None).unwrap();
        let kept_vectors = kept_vectors.as_fixed_size_list().clone();
        let queries = kept_vectors.slice(0, 50);
        let recall = recall(&loaded, &kept_vectors, &queries);
        assert!(
            recall >= 0.5,
            "recall {recall} after removing 1/{remove_every}"
        );
    }

    #[test]
    fn test_remap_replaces_removed_entry_point() {
        let (_, hnsw) = build(TOTAL);
        let batch = hnsw.to_batch().unwrap();
        let old_metadata = metadata(&batch);
        let entry_point = old_metadata.entry_point as usize;

        let new_ids = (0..TOTAL)
            .map(|old_id| match old_id.cmp(&entry_point) {
                std::cmp::Ordering::Less => Some(old_id as u32),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some(old_id as u32 - 1),
            })
            .collect::<Vec<_>>();
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        let new_metadata = metadata(&remapped);
        let loaded = HNSW::load(remapped.clone()).unwrap();

        // The replacement sits on the highest level that still has nodes.
        let top = loaded.max_level() as usize - 1;
        let ids = remapped[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let top_ids =
            &ids.values()[new_metadata.level_offsets[top]..new_metadata.level_offsets[top + 1]];
        assert!(top_ids.contains(&new_metadata.entry_point));
    }

    #[test]
    fn test_remap_removing_every_node() {
        let (_, hnsw) = build(64);
        let batch = hnsw.to_batch().unwrap();
        let remapped = remap_graph_batch(&batch, &[None; 64]).unwrap();
        assert_eq!(remapped.num_rows(), 0);
        assert!(HNSW::load(remapped).unwrap().is_empty());
    }

    #[test]
    fn test_remap_rejects_mismatched_node_count() {
        let (_, hnsw) = build(64);
        let batch = hnsw.to_batch().unwrap();
        let err = remap_graph_batch(&batch, &[Some(0); 63]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("64 nodes"), "{err}");

        // A search-projected batch lacks the distances the index file keeps.
        let projected = batch.project(&[0, 1]).unwrap();
        let new_ids = (0..64).map(Some).collect::<Vec<_>>();
        let err = remap_graph_batch(&projected, &new_ids).unwrap_err();
        assert!(err.to_string().contains(DIST_COL), "{err}");
    }
}
