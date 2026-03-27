from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


EXPECTED_DIRECTIONS = ("dir_1", "dir_2", "dir_3", "dir_4")


@dataclass(frozen=True)
class PhoneMasterRecord:
    phone_id: str
    model_id: str
    batch_id: str
    vendor_id: str
    structure_version: str
    material_version: str


@dataclass(frozen=True)
class AudioAsset:
    phone_id: str
    direction_id: str
    wav_path: str


@dataclass(frozen=True)
class LabelRecord:
    phone_id: str
    noise_type_label: str
    label_source: str


@dataclass(frozen=True)
class ValidationIssue:
    phone_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class JoinValidationReport:
    valid_phone_count: int
    issues: list[ValidationIssue] = field(default_factory=list)


def validate_joined_records(
    phones: Iterable[PhoneMasterRecord],
    audio_assets: Iterable[AudioAsset],
    labels: Iterable[LabelRecord],
) -> JoinValidationReport:
    audio_by_phone: dict[str, set[str]] = {}
    for asset in audio_assets:
        audio_by_phone.setdefault(asset.phone_id, set()).add(asset.direction_id)

    labels_by_phone: dict[str, set[str]] = {}
    for label in labels:
        labels_by_phone.setdefault(label.phone_id, set()).add(label.noise_type_label)

    issues: list[ValidationIssue] = []
    valid_phone_count = 0
    for phone in phones:
        phone_issues: list[ValidationIssue] = []
        directions = audio_by_phone.get(phone.phone_id, set())
        missing = sorted(set(EXPECTED_DIRECTIONS) - directions)
        if missing:
            phone_issues.append(
                ValidationIssue(phone.phone_id, "missing_directions", ",".join(missing))
            )

        phone_labels = sorted(labels_by_phone.get(phone.phone_id, set()))
        if len(phone_labels) > 1:
            phone_issues.append(
                ValidationIssue(phone.phone_id, "conflicting_labels", ",".join(phone_labels))
            )

        if phone_issues:
            issues.extend(phone_issues)
        else:
            valid_phone_count += 1

    return JoinValidationReport(valid_phone_count=valid_phone_count, issues=issues)
