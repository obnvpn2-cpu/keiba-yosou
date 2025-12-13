# -*- coding: utf-8 -*-
"""
src.features

競馬予想AI用の特徴量構築モジュール

このファイルを src/features/__init__.py としてコピーしてください
"""

from .feature_builder import (
    map_distance_to_cat,
    load_base_tables,
    build_features_for_race,
    build_feature_table,
    compute_global_stats,
    compute_distance_cat_stats,
    compute_recent_form,
    compute_track_condition_stats,
    compute_course_stats,
    compute_horse_weight_stats,
)

from .sqlite_store_feature import (
    create_table_feature,
    insert_feature_rows,
    get_feature_count,
    get_race_count,
    get_features_for_race,
    load_all_features,
)

__all__ = [
    # feature_builder
    "map_distance_to_cat",
    "load_base_tables",
    "build_features_for_race",
    "build_feature_table",
    "compute_global_stats",
    "compute_distance_cat_stats",
    "compute_recent_form",
    "compute_track_condition_stats",
    "compute_course_stats",
    "compute_horse_weight_stats",
    # sqlite_store_feature
    "create_table_feature",
    "insert_feature_rows",
    "get_feature_count",
    "get_race_count",
    "get_features_for_race",
    "load_all_features",
]
