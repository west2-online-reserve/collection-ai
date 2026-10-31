"""娜比娅偷吃事件的基础行为检查。"""

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import longmen_vs_nabiya as battle


def run_checks() -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        battle.display_status("长门", 100, battle.NAGATO_MAX_HP)
    assert output.getvalue().strip() == "[长门]HP: 100 / 115"

    with patch.object(battle.random, "randint", side_effect=[1, 6, 3]):
        assert battle.roll_dice(3) == 10

    assert battle.choose_nagato_action(29, 100) == "defend"
    assert battle.choose_nagato_action(30, 19) == "special"
    assert battle.choose_nagato_action(80, 80) == "attack"
    assert battle.check_critical_hit(15)
    assert battle.nabiya_ai_action(40) == "defend"
    assert battle.calculate_final_damage(20, 7) == 13
    assert battle.apply_damage(10, 20) == 0
    assert battle.is_battle_over(0, 1)
    assert battle.get_battle_result(1, 0) == "nagato"
    assert battle.get_battle_result(0, 0) == "draw"

    with (
        patch.object(battle.random, "randint", return_value=6),
        patch.object(battle.random, "random", return_value=0.0),
        redirect_stdout(io.StringIO()),
    ):
        assert battle.main_battle_loop(max_turns=10) == "nagato"


if __name__ == "__main__":
    run_checks()
    print("全部检查通过。")
