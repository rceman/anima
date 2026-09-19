from anima.physics_policy import evaluate_physics


def test_physics_policy_classifies_hard_and_soft_issues():
    report = evaluate_physics(
        {
            "body": {
                "warnings": [
                    {"code": "com_outside_support", "frame": 1, "message": "soft"},
                    {"code": "joint_limit_violation", "frame": 2, "message": "hard"},
                ]
            },
            "weapon": {
                "warnings": [
                    {"code": "weapon_direction_reversal", "frame": 3, "message": "hard"},
                ]
            },
            "system": {
                "warnings": [
                    {"code": "system_com_outside_support", "frame": 4, "message": "soft"},
                ]
            },
        }
    )

    assert not report["ok"]
    assert report["counts"]["hard"] == 2
    assert report["counts"]["warnings"] == 2
