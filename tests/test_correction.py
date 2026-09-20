from anima.correction import build_render_correction_prompt


def test_passing_report_needs_no_correction():
    prompt = build_render_correction_prompt(
        {"ok": True, "issues": []}
    )

    assert "No correction is required" in prompt


def test_failed_frames_are_grouped_into_targeted_corrections():
    report = {
        "ok": False,
        "issues": [
            {
                "frame": 3,
                "label": "attack",
                "code": "ground_contact_missing",
                "foot": "foot_l",
            },
            {
                "frame": 3,
                "label": "attack",
                "code": "sword_path_missing",
            },
            {
                "frame": 5,
                "label": "follow",
                "code": "background_drift",
            },
        ],
    }

    prompt = build_render_correction_prompt(report)

    assert "Frame 3:" in prompt
    assert "foot=foot_l" in prompt
    assert "continuous rigid sword blade" in prompt
    assert "Frame 5:" in prompt
    assert "same flat background color" in prompt
    assert "Do not change any panel that is not listed" in prompt
