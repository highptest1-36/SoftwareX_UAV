# SPDX-License-Identifier: Apache-2.0
"""Detectron2 backend (optional extra): online-TTA methods for a Faster R-CNN R50-FPN with trainable FPN adapters.

runners.py      source-only, mean teacher, Tent, WHW (predict_part / adapt_part per frame)
adapters.py     residual FPN adapters (the only trainable parameters)
heads.py        ROI heads used for self-training (explicit loss normalisation)
fingerprint.py  state hashes recorded in the trace
plugins.py      OnlineMethod plugins built from the shipped profiles (profiles/uavdt/)
"""
