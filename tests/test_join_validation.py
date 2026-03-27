from noise_relation_analyze.registry import (
    AudioAsset,
    LabelRecord,
    PhoneMasterRecord,
    ValidationIssue,
    validate_joined_records,
)


def test_validate_joined_records_flags_missing_direction_and_conflicting_label() -> None:
    phones = [
        PhoneMasterRecord(
            phone_id="P001",
            model_id="M1",
            batch_id="B1",
            vendor_id="V1",
            structure_version="S1",
            material_version="MAT1",
        )
    ]
    audio_assets = [
        AudioAsset(phone_id="P001", direction_id="dir_1", wav_path="a.wav"),
        AudioAsset(phone_id="P001", direction_id="dir_2", wav_path="b.wav"),
        AudioAsset(phone_id="P001", direction_id="dir_3", wav_path="c.wav"),
    ]
    labels = [
        LabelRecord(phone_id="P001", noise_type_label="type_1", label_source="manual"),
        LabelRecord(phone_id="P001", noise_type_label="type_2", label_source="manual"),
    ]

    report = validate_joined_records(phones, audio_assets, labels)

    assert report.valid_phone_count == 0
    assert ValidationIssue("P001", "missing_directions", "dir_4") in report.issues
    assert ValidationIssue("P001", "conflicting_labels", "type_1,type_2") in report.issues
