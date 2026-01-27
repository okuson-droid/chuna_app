import streamlit as st
import numpy as np
import pandas as pd
from functools import lru_cache
import copy

# ==========================================
# 1. 定数・設定データ定義
# ==========================================

# 各サブステータスの出現値
cr_list = [6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5]
cd_list = [12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0]
at_p_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage1_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage2_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
efficiency_list = [6.8, 7.6, 8.4, 9.2, 10.0, 10.8, 11.6, 12.4]
at_n_list = [30, 40, 50, 60, 0, 0, 0, 0]

# 出現確率ウェイト
cr_weight = [0.2429, 0.2458, 0.2023, 0.0940, 0.0840, 0.0872, 0.0269, 0.0169]
cd_weight = [0.157509, 0.274725, 0.282051, 0.076923, 0.091575, 0.080586, 0.021978, 0.014652]
at_p_weight = [0.080862, 0.086253, 0.185983, 0.226415, 0.153638, 0.194070, 0.051212, 0.021563]
damage1_weight = at_p_weight
damage2_weight = at_p_weight
efficiency_weight = at_p_weight
at_n_weight = [0.269, 0.385, 0.325, 0.021, 0, 0, 0, 0]

# 統合リスト
subst_list = [cr_list, cd_list, at_p_list, damage1_list, damage2_list, efficiency_list, at_n_list]
subst_weight = [cr_weight, cd_weight, at_p_weight, damage1_weight, damage2_weight, efficiency_weight, at_n_weight]

# コスト定義
record_list = [1100, 3025, 5775, 9875, 15875] 
TUNER_COST_PER_SLOT = 10 
INF = 10**18

# キャラプリセット定義
CHAR_SETTINGS = {
    "カスタム (手動設定)": {"coe": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1], "d1_label": "ダメアップ1", "d2_label": "ダメアップ2"},
    "今汐 (スキル特化)": {"coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05], "d1_label": "共鳴スキルダメ", "d2_label": "その他のダメ"},
    "忌炎 (重撃特化)": {"coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05], "d1_label": "重撃ダメージ", "d2_label": "共鳴解放ダメ"},
    "長離 (スキル/解放)": {"coe": [2.0, 1.0, 0.8, 1.0, 1.0, 0.2, 0.05], "d1_label": "共鳴スキルダメ", "d2_label": "共鳴解放ダメ"},
    "相里要 (解放特化)": {"coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05], "d1_label": "共鳴解放ダメ", "d2_label": "共鳴スキルダメ"},
    "ツバキ (通常特化)": {"coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.1, 0.05], "d1_label": "通常攻撃ダメ", "d2_label": "共鳴スキルダメ"},
    "ヴェリーナ/ショア (ヒーラー)": {"coe": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.1], "d1_label": "回復量(非対応)", "d2_label": "ダメアップ(不要)"},
}

# ==========================================
# 2. 関数定義 (すべて最初に定義)
# ==========================================

# --- UIヘルパー関数: 3列スライダー生成 ---
def render_substat_inputs(labels, values_list, coe, key_prefix):
    """
    labels: サブステ名リスト
    values_list: 各サブステの選択肢リスト
    coe: 係数リスト（有効判定に使用）
    key_prefix: session_stateキーの接頭辞
    Return: 入力されたサブステ値のリスト
    """
    sub_inputs = [0.0] * 7
    active_indices = [i for i, c in enumerate(coe) if c > 0]
    cols = st.columns(3)
    
    for col_idx, stat_idx in enumerate(active_indices):
        with cols[col_idx % 3]:
            # optionsには0.0を追加して未取得を選択可能にする
            sub_inputs[stat_idx] = st.select_slider(
                labels[stat_idx],
                options=[0.0] + values_list[stat_idx],
                key=f"{key_prefix}_{stat_idx}"
            )
    return sub_inputs

# --- スタイル関数 ---
def style_zeros(val):
    if isinstance(val, (int, float)) and val == 0:
        return 'color: #d0d0d0;'
    return ''

# --- 計算ロジック ---
def cal_score_now(substatus, coe):
    return sum(substatus[i] * coe[i] for i in range(7))

def possible_next_states(substatus, t, coe):
    next_states = []
    sub_list = list(substatus)
    empty_idx = [i for i, v in enumerate(sub_list) if v == 0]
    
    n = 8 + t
    m = 8 - len(empty_idx) + t
    if n == 0: return []
    p_choose_type = 1 / n
    
    # 1. 不要/未取得枠
    next_states.append((tuple(sub_list), p_choose_type * m))
    
    for i in empty_idx:
        # 2. 係数0
        if coe[i] == 0:
            temp = list(sub_list)
            temp[i] = 1 
            next_states.append((tuple(temp), p_choose_type))
            continue
        # 3. 有効サブステ
        for val, w in zip(subst_list[i], subst_weight[i]):
            temp = list(sub_list)
            temp[i] = val
            next_states.append((tuple(temp), p_choose_type * w))
    return next_states

@lru_cache(maxsize=None)
def expected_resources(substatus, depth, target_score, ave_chuna, coe):
    score_now = cal_score_now(substatus, coe)
    if score_now >= target_score:
        return 0.0, 0.0, 1.0, True
    if depth == 5:
        return INF, INF, 0.0, False

    step_tuner = TUNER_COST_PER_SLOT
    step_record = record_list[depth]
    t_remaining = 5 - (depth + 1)
    
    total_prob = 0.0
    weighted_future_tuner = 0.0
    weighted_future_record = 0.0

    for sub_next, prob_trans in possible_next_states(substatus, t_remaining, coe):
        e_tuner, e_record, prob_success, ok = expected_resources(sub_next, depth + 1, target_score, ave_chuna, coe)
        if not ok or e_tuner > ave_chuna:
            continue
        prob_combined = prob_trans * prob_success
        total_prob += prob_combined
        weighted_future_tuner += prob_combined * e_tuner
        weighted_future_record += prob_combined * e_record

    if total_prob == 0:
        return INF, INF, 0.0, False

    final_expected_tuner = (step_tuner + weighted_future_tuner) / total_prob
    final_expected_record = (step_record + weighted_future_record) / total_prob
    return final_expected_tuner, final_expected_record, total_prob, True

def cal_ave_chuna_fast(target_score, times, substatus, ave_chuna, coe):
    val, _, _, ok = expected_resources(tuple(substatus), times, target_score, ave_chuna, tuple(coe))
    return val if ok else INF

def cal_max_score(coe):
    arr = [coe[i] * subst_list[i][7] for i in range(len(coe)) if coe[i] > 0]
    arr.sort(reverse=True)
    return 53.6 + arr[0] + arr[1] if len(arr) >= 2 else 60.0

def cal_min_chuna(score, coe):
    low, high, ans = 1.0, 300000.0, 300000.0
    coe_t = tuple(coe)
    for _ in range(30):
        mid = (low + high) / 2
        res_tuner, _, _, ok = expected_resources(tuple([0]*7), 0, score, mid, coe_t)
        if ok and res_tuner <= mid:
            ans = high = mid
        else:
            low = mid
    return ans

def cal_max_score_by_chuna(chuna_limit, coe, max_s):
    low, high, ans = 1.0, max_s, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        if cal_min_chuna(mid, coe) <= chuna_limit:
            ans = low = mid
        else:
            high = mid
    return ans

def judge_continue(score, times, substatus, ave_chuna, coe):
    e_tuner, _, _, ok = expected_resources(tuple(substatus), times, score, ave_chuna, tuple(coe))
    if not ok: return INF, "達成不可能"
    if e_tuner <= ave_chuna: return e_tuner, "強化推奨"
    elif e_tuner <= ave_chuna * 1.2: return e_tuner, "続行可能"
    return e_tuner, "強化非推奨"

def judge_continue_all(score, times, ave_chuna, coe):
    results = []
    subst_list0 = copy.deepcopy(subst_list)
    for i in range(7): subst_list0[i].insert(0, 0)
    memory = np.zeros((1, 7)) + 8
    
    if times == 0:
        ch = cal_ave_chuna_fast(score, 0, [0]*7, ave_chuna, coe)
        if ch <= ave_chuna: results.append({"substatus": [0]*7, "chuna": ch, "score": 0})
        return results

    if times >= 1:
        for i in range(7):
            if coe[i] == 0: continue
            for j in range(8):
                if i == 6 and j >= 4: break
                sub = [0]*7; sub[i] = subst_list0[i][j+1]
                ch = cal_ave_chuna_fast(score, times, sub, ave_chuna, coe)
                if ch <= ave_chuna:
                    results.append({"substatus": sub, "chuna": ch, "score": cal_score_now(sub, coe)})
                    memory = np.append(memory, [[0]*7], axis=0); memory[-1, i] = j + 1
                    break
    if times == 1: return results

    if times >= 2:
        for i in range(6):
            if coe[i] == 0: continue
            for j in range(8):
                st_ = np.zeros(7); st_[i] = j + 1
                if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: break
                for k in range(i+1, 7):
                    if coe[k] == 0: continue
                    for l in range(8):
                        if k == 6 and l >= 4: break
                        st_[k] = l + 1
                        if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0:
                            st_[k] = 0; break
                        st_[k] = 0; sub = [0]*7; sub[i] = subst_list0[i][j+1]; sub[k] = subst_list0[k][l+1]
                        ch = cal_ave_chuna_fast(score, times, sub, ave_chuna, coe)
                        if ch <= ave_chuna:
                            results.append({"substatus": sub, "chuna": ch, "score": cal_score_now(sub, coe)})
                            memory = np.append(memory, [[0]*7], axis=0); memory[-1, i] = j+1; memory[-1, k] = l+1
                            break
    if times == 2:




    




