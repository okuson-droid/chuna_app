import streamlit as st
import numpy as np
import pandas as pd
from functools import lru_cache
import copy

# ==========================================
# 1. 定数・データ定義
# ==========================================

# 各サブステータスの出現値リスト
cr_list = [6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5]
cd_list = [12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0]
at_p_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage1_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage2_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
efficiency_list = [6.8, 7.6, 8.4, 9.2, 10.0, 10.8, 11.6, 12.4]
at_n_list = [30, 40, 50, 60, 0, 0, 0, 0]

# 各値の出現確率ウェイト
cr_weight = [0.2429, 0.2458, 0.2023, 0.0940, 0.0840, 0.0872, 0.0269, 0.0169]
cd_weight = [0.157509, 0.274725, 0.282051, 0.076923, 0.091575, 0.080586, 0.021978, 0.014652]
at_p_weight = [0.080862, 0.086253, 0.185983, 0.226415, 0.153638, 0.194070, 0.051212, 0.021563]
damage1_weight = at_p_weight
damage2_weight = at_p_weight
efficiency_weight = at_p_weight
at_n_weight = [0.269, 0.385, 0.325, 0.021, 0, 0, 0, 0]

subst_list = [cr_list, cd_list, at_p_list, damage1_list, damage2_list, efficiency_list, at_n_list]
subst_weight = [cr_weight, cd_weight, at_p_weight, damage1_weight, damage2_weight, efficiency_weight, at_n_weight]

# コスト関連データ
record_list = [1100, 3025, 5775, 9875, 15875] 
TUNER_COST_PER_SLOT = 10 

INF = 10**18

# ==========================================
# 2. 計算ロジック関数群
# ==========================================

def cal_score_now(substatus, coe):
    """現在のスコアを計算"""
    s = 0
    for i in range(7):
        s += substatus[i] * coe[i]
    return s

def possible_next_states(substatus, t, coe):
    """次の状態遷移と確率を列挙"""
    next_states = []
    sub_list = list(substatus)
    empty_idx = [i for i, v in enumerate(sub_list) if v == 0]
    
    n = 8 + t
    m = 8 - len(empty_idx) + t
    
    if n == 0: return []

    p_choose_type = 1 / n
    
    # A. 不要サブステ(またはハズレ枠)
    next_states.append(
        (tuple(sub_list), p_choose_type * m)
    )

    for i in empty_idx:
        # B. 係数0の有効リスト値
        if coe[i] == 0:
            temp = list(sub_list)
            temp[i] = 1 
            next_states.append(
                (tuple(temp), p_choose_type)
            )
            continue

        # C. 有効サブステ
        for val, w in zip(subst_list[i], subst_weight[i]):
            temp = list(sub_list)
            temp[i] = val
            next_states.append(
                (tuple(temp), p_choose_type * w)
            )

    return next_states

@lru_cache(maxsize=None)
def expected_resources(substatus, depth, target_score, ave_chuna, coe):
    """正確なリソース消費期待値を計算するDP関数"""
    score_now = cal_score_now(substatus, coe)

    # 目標達成
    if score_now >= target_score:
        return 0.0, 0.0, 1.0, True

    # 全スロット開放済みだが目標未達
    if depth == 5:
        return INF, INF, 0.0, False

    step_tuner = TUNER_COST_PER_SLOT
    step_record = record_list[depth]
    t_remaining = 5 - (depth + 1)
    
    total_prob = 0.0
    weighted_future_tuner = 0.0
    weighted_future_record = 0.0

    for sub_next, prob_trans in possible_next_states(substatus, t_remaining, coe):
        e_tuner, e_record, prob_success, ok = expected_resources(
            sub_next,
            depth + 1,
            target_score,
            ave_chuna,
            coe
        )

        if not ok:
            continue

        # 損切り判定
        if e_tuner > ave_chuna:
            continue

        total_prob += prob_trans * prob_success
        weighted_future_tuner += prob_trans * prob_success * e_tuner
        weighted_future_record += prob_trans * prob_success * e_record

    if total_prob == 0:
        return INF, INF, 0.0, False

    final_expected_tuner = (step_tuner + weighted_future_tuner) / total_prob
    final_expected_record = (step_record + weighted_future_record) / total_prob
    
    return final_expected_tuner, final_expected_record, total_prob, True

def cal_max_score(coe):
    """理論最大スコア計算"""
    arr = [coe[i] * subst_list[i][7] for i in range(len(coe)) if coe[i] > 0]
    arr.sort(reverse=True)
    if len(arr) >= 2:
        return 53.6 + arr[0] + arr[1] 
    return 60.0

def cal_min_chuna(score, coe):
    """目標スコアに必要な最小チュナ期待値を二分探索"""
    low = 1
    high = 300000
    ans = high
    coe_t = tuple(coe)
    
    for _ in range(30):
        mid = (low + high) / 2
        res_tuner, _, _, ok = expected_resources(tuple([0]*7), 0, score, mid, coe_t)
        
        if ok and res_tuner <= mid:
            ans = mid
            high = mid
        else:
            low = mid
    return ans

def cal_max_score_by_chuna(chuna_limit, coe, max_s):
    """指定チュナで狙える最大スコアを二分探索"""
    low = 1.0
    high = max_s
    ans = low
    for _ in range(20):
        mid = (low + high) / 2
        req = cal_min_chuna(mid, coe)
        if req <= chuna_limit:
            ans = mid
            low = mid
        else:
            high = mid
    return ans

def judge_continue(score, times, substatus, ave_chuna, coe):
    """続行判定"""
    substatus_t = tuple(substatus)
    coe_t = tuple(coe)
    e_tuner, _, _, ok = expected_resources(substatus_t, times, score, ave_chuna, coe_t)
    
    if not ok: return INF, "達成不可能"
    
    if e_tuner <= ave_chuna:
        return e_tuner, "強化推奨"
    elif e_tuner <= ave_chuna * 1.2:
        return e_tuner, "続行可能"
    else:
        return e_tuner, "強化非推奨"

# ==========================================
# 3. Streamlit UI
# ==========================================

st.set_page_config(page_title="音骸厳選計算ツール", layout="wide")

# --- プリセットデータの定義 ---
CHAR_SETTINGS = {
    "カスタム (手動設定)": {
        "coe": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1],
        "d1_label": "ダメアップ1",
        "d2_label": "ダメアップ2"
    },
    "今汐 (スキル特化)": {
        "coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05],
        "d1_label": "共鳴スキルダメ",
        "d2_label": "その他のダメ"
    },
    "忌炎 (重撃特化)": {
        "coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05],
        "d1_label": "重撃ダメージ",
        "d2_label": "共鳴解放ダメ"
    },
    "長離 (スキル/解放)": {
        "coe": [2.0, 1.0, 0.8, 1.0, 1.0, 0.2, 0.05],
        "d1_label": "共鳴スキルダメ",
        "d2_label": "共鳴解放ダメ"
    },
    "相里要 (解放特化)": {
        "coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.2, 0.05],
        "d1_label": "共鳴解放ダメ",
        "d2_label": "共鳴スキルダメ"
    },
    "ツバキ (通常特化)": {
        "coe": [2.0, 1.0, 0.8, 1.0, 0.0, 0.1, 0.05],
        "d1_label": "通常攻撃ダメ",
        "d2_label": "共鳴スキルダメ"
    },
    "ヴェリーナ/ショア (ヒーラー)": {
        "coe": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.1],
        "d1_label": "回復量(非対応)",
        "d2_label": "ダメアップ(不要)"
    },
}

def update_presets():
    selected = st.session_state["char_selector"]
    vals = CHAR_SETTINGS[selected]["coe"]
    st.session_state["ni_cr"] = vals[0]
    st.session_state["ni_cd"] = vals[1]
    st.session_state["ni_atk"] = vals[2]
    st.session_state["ni_d1"] = vals[3]
    st.session_state["ni_d2"] = vals[4]
    st.session_state["ni_er"] = vals[5]
    st.session_state["ni_flat"] = vals[6]

with st.sidebar:
    st.header("1. スコア計算の設定")
    selected_char = st.selectbox(
        "キャラ・プリセット選択",
        options=list(CHAR_SETTINGS.keys()),
        key="char_selector",
        on_change=update_presets
    )
    setting = CHAR_SETTINGS[selected_char]
    
    if "ni_cr" not in st.session_state:
        init_vals = setting["coe"]
        st.session_state["ni_cr"] = init_vals[0]
        st.session_state["ni_cd"] = init_vals[1]
        st.session_state["ni_atk"] = init_vals[2]
        st.session_state["ni_d1"] = init_vals[3]
        st.session_state["ni_d2"] = init_vals[4]
        st.session_state["ni_er"] = init_vals[5]
        st.session_state["ni_flat"] = init_vals[6]

    coe_input = [0.0] * 7
    coe_input[0] = st.number_input("クリティカル", step=0.1, format="%.2f", key="ni_cr")
    coe_input[1] = st.number_input("クリダメ", step=0.1, format="%.2f", key="ni_cd")
    coe_input[2] = st.number_input("攻撃力％", step=0.1, format="%.2f", key="ni_atk")
    st.divider()
    coe_input[3] = st.number_input(setting["d1_label"], step=0.1, format="%.2f", key="ni_d1")
    coe_input[4] = st.number_input(setting["d2_label"], step=0.1, format="%.2f", key="ni_d2")
    st.divider()
    coe_input[5] = st.number_input("共鳴効率", step=0.1, format="%.2f", key="ni_er")
    coe_input[6] = st.number_input("攻撃実数", step=0.01, format="%.2f", key="ni_flat")
    coe = coe_input

# ==========================================
# 4. メインエリア
# ==========================================

current_sub_names = [
    "クリティカル", "クリダメ", "攻撃力％",
    setting["d1_label"], setting["d2_label"],
    "共鳴効率", "攻撃実数"
]

st.title("鳴潮 音骸厳選計算ツール")

tab1, tab2, tab3 = st.tabs(["① 目標設定", "② 続行判定", "④ 最小ライン一覧"])

if 'target_score' not in st.session_state:
    st.session_state['target_score'] = 25.0
if 'ave_chuna' not in st.session_state:
    st.session_state['ave_chuna'] = 100.0

# --- TAB 1: 目標設定 ---
with tab1:
    st.header("目標スコアの算出")
    st.info("許容コストから目標スコアを逆算、または目標スコアから正確な消費期待値を算出します。")
    
    col1, col2 = st.columns(2)
    with col1:
        limit_chuna = st.number_input("使用可能なチュナの上限（期待値）", value=500, step=100)
        if st.button("チュナ上限から計算"):
            with st.spinner("計算中..."):
                max_s = cal_max_score(coe)
                calc_score = cal_max_score_by_chuna(limit_chuna, coe, max_s)
                st.session_state['target_score'] = calc_score
                
                # 正確なリソース計算 (depth=0から)
                tuner_req = cal_min_chuna(calc_score, coe)
                c, r, prob, ok = expected_resources(tuple([0]*7), 0, calc_score, tuner_req, tuple(coe))
                
                st.session_state['ave_chuna'] = c
                st.session_state['res_c'] = c
                st.session_state['res_r'] = r
                st.session_state['res_b'] = 1 / prob if prob > 0 else INF
                
            st.success(f"推奨目標スコア: **{calc_score:.2f}**")
            
    with col2:
        val = st.number_input("目標スコアを直接入力", value=st.session_state['target_score'], step=1.0)
        if st.button("スコア入力で再計算"):
            st.session_state['target_score'] = val
            with st.spinner("計算中..."):
                tuner_req = cal_min_chuna(val, coe)
                c, r, prob, ok = expected_resources(tuple([0]*7), 0, val, tuner_req, tuple(coe))
                
                st.session_state['ave_chuna'] = c
                st.session_state['res_c'] = c
                st.session_state['res_r'] = r
                st.session_state['res_b'] = 1 / prob if prob > 0 else INF
                st.toast("計算完了")

    st.divider()
    
    # リソース消費量の表示
    if 'res_c' in st.session_state:
        st.subheader("想定リソース消費量（期待値）")
        st.caption("※元のアルゴリズムに基づき、record_listの値を正確に使用して算出しています。")
        m1, m2, m3 = st.columns(3)
        m1.metric("チュナ消費量", f"{int(st.session_state['res_c']):,}")
        m2.metric("レコード消費量", f"{int(st.session_state['res_r']):,}")
        
        b_val = st.session_state['res_b']
        b_str = f"{b_val:.1f} 個" if b_val < 10000 else "∞"
        m3.metric("音骸素体消費量", b_str)
    else:
        st.caption("※計算ボタンを押すとここに消費量が表示されます")


# --- TAB 2: 続行判定 ---
with tab2:
    st.header("強化続行・撤退の判定")
    st.markdown(f"目標スコア **{st.session_state['target_score']:.2f}** を目指す場合の判定を行います。")
    
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        st.subheader("現在の音骸ステータス")
        current_times = st.slider("現在のサブステ開放数（強化回数）", 0, 4, 1)
        current_sub = [0.0] * 7
        active_indices = [i for i, c in enumerate(coe) if c > 0]
        
        input_count = 0
        for i in active_indices:
            vals = [0.0] + subst_list[i]
            val = st.select_slider(
                f"{current_sub_names[i]}",
                options=vals,
                key=f"slider_{i}"
            )
            current_sub[i] = val
            if val > 0: input_count += 1
            
        if input_count > current_times:
            st.error(f"入力数({input_count})が開放数({current_times})を超えています。")

    with col_res:
        st.subheader("判定結果")
        current_score = cal_score_now(current_sub, coe)
        st.metric("現在のスコア", f"{current_score:.2f}")
        
        if st.button("判定実行", type="primary"):
            if input_count > current_times:
                st.error("入力値を確認してください")
            else:
                cost, msg = judge_continue(
                    st.session_state['target_score'],
                    current_times,
                    current_sub,
                    st.session_state['ave_chuna'],
                    coe
                )
                
                if msg == "強化推奨":
                    st.success(f"## {msg}")
                    st.write("このまま強化するのが期待値的に**お得**です。")
                elif "続行可能" in msg:
                    st.warning(f"## {msg}")
                else:
                    st.error(f"## {msg}")
                    st.write("これ以上強化すると期待値的に**損**です。")
                
                if cost < INF:
                    diff = st.session_state['ave_chuna'] - cost
                    st.write(f"ゴールまでの期待コスト: **{int(cost)}** チュナ")
                    if diff > 0:
                        st.caption(f"新品より {int(diff)} チュナ節約の見込み")


# --- TAB 3: 最小ライン一覧 (修正完了版) ---
with tab3:
    st.header("これ以上強化していい最小ライン一覧")
    st.info("「このサブステが付いたら強化を続けても良い」という**最低ライン**の組み合わせを表示します。")
    st.caption("※より強力な上位互換（例：クリ6.3でOKな場合のクリ10.5など）は、この表では省略されます。")

    if 'target_score' not in st.session_state:
        st.error("先にタブ①で目標スコアを計算してください")
    else:
        search_times = st.selectbox("検索する強化回数", [1, 2, 3, 4], help="回数が多いと計算に時間がかかります")
        
        if st.button("一覧を生成"):
            with st.spinner("探索中..."):
                valid_rows = []
                ave = st.session_state['ave_chuna']
                score_target = st.session_state['target_score']
                
                valid_idx = [i for i, c in enumerate(coe) if c > 0]
                
                # --- 1. 全探索 (指定された強化回数分の数値を埋める) ---
                def search_combos(idx_in_valid, current_state, count):
                    # 規定回数分埋まったら判定
                    if count == 0:
                        cost, msg = judge_continue(score_target, search_times, current_state, ave, coe)
                        if "非推奨" not in msg:
                            row = {}
                            row["_raw_state"] = list(current_state) 
                            row["スコア"] = cal_score_now(current_state, coe)
                            row["消費チュナ"] = int(cost)
                            for i in range(7):
                                row[current_sub_names[i]] = current_state[i]
                            valid_rows.append(row)
                        return

                    # インデックス切れ
                    if idx_in_valid >= len(valid_idx):
                        return

                    real_idx = valid_idx[idx_in_valid]
                    
                    # 1. このサブステ種類を選ばない（スキップ）
                    search_combos(idx_in_valid + 1, list(current_state), count)
                    
                    # 2. このサブステ種類を選ぶ（値を入れる）
                    #    ※ここで0は選ばない (countを消費するのに値が0だと意味が変わるため)
                    for val in subst_list[real_idx]:
                        if val > 0:
                            new_state = list(current_state)
                            new_state[real_idx] = val
                            # 次のサブステへ（重複なし前提）
                            search_combos(idx_in_valid + 1, new_state, count - 1)

                search_combos(0, [0.0]*7, search_times)
                
                if not valid_rows:
                    st.warning("条件を満たす組み合わせが見つかりませんでした。（すべて非推奨です）")
                else:
                    df = pd.DataFrame(valid_rows)
                    
                    # --- 2. フィルタリング (上位互換の削除) ---
                    # スコア順に並べて、「より弱い構成でOKなら、強い構成はリストから消す」
                    df = df.sort_values("スコア")
                    final_indices = []
                    rows_data = df.to_dict('records')
                    active_indices = [i for i, c in enumerate(coe) if c > 0]
                    
                    for i, current in enumerate(rows_data):
                        is_redundant = False
                        current_state = current["_raw_state"]
                        
                        for kept_idx in final_indices:
                            kept = rows_data[kept_idx]
                            kept_state = kept["_raw_state"]
                            
                            # 条件: すべての有効ステータスにおいて kept <= current
                            # 値が0の部分も含めて「以下」であれば、keptが下位互換（最小ライン）となる
                            # 例: kept=[クリ6.3, 他0], current=[クリ6.3, クリダメ12.6]
                            #     6.3<=6.3 (OK), 0<=12.6 (OK) -> currentは冗長なので削除
                            is_dominated = True
                            for idx in active_indices:
                                if kept_state[idx] > current_state[idx]:
                                    is_dominated = False
                                    break
                            
                            if is_dominated:
                                is_redundant = True
                                break
                        
                        if not is_redundant:
                            final_indices.append(i)
                            
                    df_filtered = df.iloc[final_indices].reset_index(drop=True)
                    
                    # --- 3. 表示用整形 ---
                    # カラム順序の固定
                    base_order = [
                        "クリティカル", "クリダメ", "攻撃力％",
                        setting["d1_label"], setting["d2_label"],
                        "共鳴効率", "攻撃実数"
                    ]
                    # 係数が0の列は除外
                    active_cols = [name for i, name in enumerate(base_order) if coe[i] > 0]
                    final_cols = active_cols + ["スコア", "消費チュナ"]
                    
                    df_display = df_filtered[final_cols]
                    
                    # --- 4. スタイルとフォーマット ---
                    # 0の値を目立たなくするスタイル
                    def style_zeros(val):
                        if isinstance(val, (int, float)) and val == 0:
                            return 'color: #d0d0d0; font-weight: 300;'
                        return ''
                    
                    # 数値フォーマットの定義 (すべて小数点1桁)
                    col_config = {}
                    for col in active_cols:
                        col_config[col] = st.column_config.NumberColumn(format="%.1f")
                    col_config["スコア"] = st.column_config.NumberColumn(format="%.1f")
                    col_config["消費チュナ"] = st.column_config.NumberColumn(format="%d") # チュナは整数

                    st.write(f"**強化回数 {search_times}回目** の続行可能最小ライン ({len(df_display)}件)")
                    
                    st.dataframe(
                        df_display.style.map(style_zeros),
                        column_config=col_config,
                        use_container_width=True,
                        hide_index=True
                    )



    




