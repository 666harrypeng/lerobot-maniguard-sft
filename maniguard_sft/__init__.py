"""ManiGuard SmolVLA SFT layer (vendored into the lerobot-maniguard-sft fork).

Thin, self-contained layer around LeRobot's native SmolVLA policy: fine-tuned via
the upstream ``lerobot-train`` CLI (this fork pins **lerobot v0.5.1**, which supports
multi-GPU training via ``accelerate``). SmolVLA has no config registry / embodiment
registration — ``lerobot-train`` derives the policy's input/output features straight
from the dataset's standard ``observation.*`` / ``action`` keys. So the only
ManiGuard-side artifact is the embodiment *contract* (``embodiment``): how the 5-cam
datagen LeRobot export (flat keys ``image_*`` / ``state`` / ``actions``) maps into the
2-cam, standard-keyed dataset SmolVLA expects. Runnable tooling: ``tools/smolvla_sft``.
"""
