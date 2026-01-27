import streamlit as st
import numpy as np
import pandas as pd
from functools import lru_cache
import copy

# ==========================================
# 1. 定数・データ定義
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
# 2. 計算ロジック関数群
# ==========================================

def cal_score_now(substatus, coe):
    """現在のスコア算出"""
    return sum(substatus[i] * coe[i] for i in range(7))

def possible_next_states(substatus, t, coe):
    """次の状態遷移と確率を生成"""
    next_states = []
    # タプルだと操作しづらいためリスト化
    sub_list = list(substatus)
    empty_idx = [i for i, v in enumerate(sub_list) if v == 0]
    
    n = 8 + t
    m = 8 - len(empty_idx) + t
    
    if n == 0: return []
    p_choose_type = 1 / n
    
    # 1. 不要サブステ(または未取得枠)が選ばれるケース
    next_states.append((tuple(sub_list), p_choose_type * m))
    
    for i in empty_idx:
        # 2. 係数0（スコア対象外）のサブステが選ばれるケース
        if coe[i] == 0:
            temp = list(sub_list)
            temp[i] = 1 # 取得済みフラグ
            next_states.append((tuple(temp), p_choose_type))
            continue
        
        # 3. 有効サブステが選ばれるケース（値の抽選）
        for val, w in zip(subst_list[i], subst_weight[i]):
            temp = list(sub_list)
            temp[i] = val
            next_states.append((tuple(temp), p_choose_type * w))
            
    return next_states

# 最適化: maxsize=None でキャッシュ全保持（状態数は有限のためメモリ圧迫より速度を優先）
@lru_cache(maxsize=None)
def expected_resources(substatus, depth, target_score, ave_chuna, coe):
    """DPによる期待リソース計算"""
    score_now = cal_score_now(substatus, coe)
    
    # 成功
    if score_now >= target_score:
        return 0.0, 0.0, 1.0, True
    # 失敗（回数切れ）
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
        
        # 損切りラインを超えている、または達成不可能な場合はスキップ
        if not ok or e_tuner > ave_chuna:
            continue
            
        prob_combined = prob_trans * prob_success
        total_prob += prob_combined
        weighted_future_tuner += prob_combined * e_tuner
        weighted_future_record += prob_combined * e_record

    if total_prob == 0:
        return INF, INF, 0.0, False

    # 期待値 = (今回コスト + 将来の加重平均コスト) / 成功確率
    final_expected_tuner = (step_tuner + weighted_future_tuner) / total_prob
    final_expected_record = (step_record + weighted_future_record) / total_prob
    
    return final_expected_tuner, final_expected_record, total_prob, True

def cal_ave_chuna_fast(target_score, times, substatus, ave_chuna, coe):
    """judge_continue_all用のラッパー"""
    val, _, _, ok = expected_resources(tuple(substatus), times, target_score, ave_chuna, tuple(coe))
    return val if ok else INF

def cal_max_score(coe):
    """理論最大スコア近似値"""
    arr = [coe[i] * subst_list[i][7] for i in range(len(coe)) if coe[i] > 0]
    arr.sort(reverse=True)
    return 53.6 + arr[0] + arr[1] if len(arr) >= 2 else 60.0

def cal_min_chuna(score, coe):
    """目標スコアに必要な最小チュナ期待値（二分探索）"""
    low, high, ans = 1.0, 300000.0, 300000.0
    coe_t = tuple(coe)
    # 探索精度
    for _ in range(30):
        mid = (low + high) / 2
        res_tuner, _, _, ok = expected_resources(tuple([0]*7), 0, score, mid, coe_t)
        if ok and res_tuner <= mid:
            ans = high = mid
        else:
            low = mid
    return ans

def cal_max_score_by_chuna(chuna_limit, coe, max_s):
    """指定コストで達成可能な最大スコア（二分探索）"""
    low, high, ans = 1.0, max_s, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        if cal_min_chuna(mid, coe) <= chuna_limit:
            ans = low = mid
        else:
            high = mid
    return ans

def judge_continue(score, times, substatus, ave_chuna, coe):
    """続行判定"""
    e_tuner, _, _, ok = expected_resources(tuple(substatus), times, score, ave_chuna, tuple(coe))
    if not ok: return INF, "達成不可能"
    if e_tuner <= ave_chuna: return e_tuner, "強化推奨"
    elif e_tuner <= ave_chuna * 1.2: return e_tuner, "続行可能"
    return e_tuner, "強化非推奨"

def judge_continue_all(score, times, ave_chuna, coe):
    """
    全探索による続行ライン算出（仕様維持）
    """
    results = []
    subst_list0 = copy.deepcopy(subst_list)
    for i in range(7): subst_list0[i].insert(0, 0)
    
    # 枝刈り用メモリ
    memory = np.zeros((1, 7)) + 8
    
    # times=0
    if times == 0:
        ch = cal_ave_chuna_fast(score, 0, [0]*7, ave_chuna, coe)
        if ch <= ave_chuna: results.append({"substatus": [0]*7, "chuna": ch, "score": 0})
        return results

    # times=1 loop
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

    # times=2 loop
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
                        if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[k] = 0; break
                        st_[k] = 0; sub = [0]*7; sub[i] = subst_list0[i][j+1]; sub[k] = subst_list0[k][l+1]
                        ch = cal_ave_chuna_fast(score, times, sub, ave_chuna, coe)
                        if ch <= ave_chuna:
                            results.append({"substatus": sub, "chuna": ch, "score": cal_score_now(sub, coe)})
                            memory = np.append(memory, [[0]*7], axis=0); memory[-1, i] = j+1; memory[-1, k] = l+1
                            break
    if times == 2: return results

    # times=3 loop
    if times >= 3:
        for i in range(5):
            if coe[i] == 0: continue
            for j in range(8):
                st_ = np.zeros(7); st_[i] = j + 1
                if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: break
                for k in range(i+1, 6):
                    if coe[k] == 0: continue
                    for l in range(8):
                        if k == 6 and l >= 4: break
                        st_[k] = l + 1
                        if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[k] = 0; break
                        for m in range(k+1, 7):
                            if coe[m] == 0: continue
                            for n in range(8):
                                if m == 6 and n >= 4: break
                                st_[m] = n + 1
                                if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[m] = 0; break
                                st_[m] = 0; sub = [0]*7; sub[i] = subst_list0[i][j+1]; sub[k] = subst_list0[k][l+1]; sub[m] = subst_list0[m][n+1]
                                ch = cal_ave_chuna_fast(score, times, sub, ave_chuna, coe)
                                if ch <= ave_chuna:
                                    results.append({"substatus": sub, "chuna": ch, "score": cal_score_now(sub, coe)})
                                    memory = np.append(memory, [[0]*7], axis=0); memory[-1, i] = j+1; memory[-1, k] = l+1; memory[-1, m] = n+1
                                    break
                        st_[k] = 0
    if times == 3: return results

    # times=4 loop
    if times >= 4:
        for i in range(4):
            if coe[i] == 0: continue
            for j in range(8):
                st_ = np.zeros(7); st_[i] = j+1
                if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: break
                for k in range(i+1, 5):
                    if coe[k] == 0: continue
                    for l in range(8):
                        st_[k] = l+1
                        if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[k] = 0; break
                        for m in range(k+1, 6):
                            if coe[m] == 0: continue
                            for n in range(8):
                                st_[m] = n+1
                                if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[m] = 0; break
                                for o in range(m+1, 7):
                                    if coe[o] == 0: continue
                                    for p in range(8):
                                        if o == 6 and p >= 4: break
                                        st_[o] = p+1
                                        if np.count_nonzero(np.all(memory <= st_, axis=1)) > 0: st_[o] = 0; break
                                        st_[o] = 0; sub = [0]*7; sub[i] = subst_list0[i][j+1]; sub[k] = subst_list0[k][l+1]; sub[m] = subst_list0[m][n+1]; sub[o] = subst_list0[o][p+1]
                                        ch = cal_ave_chuna_fast(score, times, sub, ave_chuna, coe)
                                        if ch <= ave_chuna:
                                            results.append({"substatus": sub, "chuna": ch, "score": cal_score_now(sub, coe)})
                                            memory = np.append(memory, [[0]*7], axis=0); memory[-1, i] = j+1; memory[-1, k] = l+1; memory[-1, m] = n+1; memory[-1, o] = p+1
                                            break
                                st_[m] = 0
                        st_[k] = 0
    return results

# ==========================================
# 3. Streamlit UI
# ==========================================

st.set_page_config(page_title="音骸厳選計算ツール", layout="wide")

# 初期化を一括で行う
if 'target_score' not in st.session_state: st.session_state['target_score'] = 25
if 'ave_chuna' not in st.session_state: st.session_state['ave_chuna'] = 100.0

def update_presets():
    sel = st.session_state["char_selector"]
    vals = CHAR_SETTINGS[sel]["coe"]
    keys = ["ni_cr", "ni_cd", "ni_atk", "ni_d1", "ni_d2", "ni_er", "ni_flat"]
    for i, k in enumerate(keys):
        st.session_state[k] = vals[i]

# UIヘルパー関数: 3列スライダー生成（コード重複削減）
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
            sub_inputs[stat_idx] = st.select_slider(
                labels[stat_idx],
                options=[0.0] + values_list[stat_idx],
                key=f"{key_prefix}_{stat_idx}"
            )
    return sub_inputs

# スタイル関数
def style_zeros(val):
    if isinstance(val, (int, float)) and val == 0:
        return 'color: #d0d0d0;'
    return ''

# --- サイドバー ---
with st.sidebar:
    st.header("1. スコア計算の設定")
    sel_char = st.selectbox("キャラ・プリセット選択", options=list(CHAR_SETTINGS.keys()), key="char_selector", on_change=update_presets)
    set_ = CHAR_SETTINGS[sel_char]
    
    # 初回用の値セット
    if "ni_cr" not in st.session_state:
        update_presets()
        
    labels = ["クリティカル", "クリダメ", "攻撃力％", set_["d1_label"], set_["d2_label"], "共鳴効率", "攻撃実数"]
    keys = ["ni_cr", "ni_cd", "ni_atk", "ni_d1", "ni_d2", "ni_er", "ni_flat"]
    coe = []
    
    # 係数入力フォーム生成
    for i, (lbl, k) in enumerate(zip(labels, keys)):
        step_val = 0.01 if i == 6 else 0.1 # 攻撃実数だけstep細かい
        if i == 3 or i == 4: st.divider() # 区切り線
        coe.append(st.number_input(lbl, step=step_val, format="%.2f", key=k))
    
    # 5,6番目の前にも区切り
    st.divider()

current_sub_names = labels # 現在の表示ラベルリスト

# --- メイン画面 ---
st.title("鳴潮 音骸厳選計算ツール")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 目標設定", "② 続行判定", "③ 最小ライン", "④ スコア計算(単体)", "⑤ 5連音骸管理"])

# --- Tab 1: 目標設定 ---
with tab1:
    st.header("目標スコアの算出")
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("使用可能なチュナの上限（期待値）", value=500.0, step=100.0)
        if st.button("チュナ上限から計算"):
            with st.spinner("計算中..."):
                sc = cal_max_score_by_chuna(limit, coe, cal_max_score(coe))
                sc_int = int(sc)
                st.session_state['target_score'] = sc_int
                st.session_state['res_c'], st.session_state['res_r'], pr, _ = expected_resources(tuple([0]*7), 0, sc_int, cal_min_chuna(sc_int, coe), tuple(coe))
                st.session_state['ave_chuna'] = st.session_state['res_c']
                st.session_state['res_b'] = 1 / pr if pr > 0 else INF
                st.success(f"推奨目標スコア: **{sc_int}**")
    with col2:
        val = st.number_input("目標スコアを直接入力", value=int(st.session_state['target_score']), step=1, format="%d")
        if st.button("スコア入力で再計算"):
            st.session_state['target_score'] = val
            with st.spinner("計算中..."):
                min_c = cal_min_chuna(val, coe)
                st.session_state['res_c'], st.session_state['res_r'], pr, _ = expected_resources(tuple([0]*7), 0, val, min_c, tuple(coe))
                st.session_state['ave_chuna'] = st.session_state['res_c']
                st.session_state['res_b'] = 1 / pr if pr > 0 else INF
    
    if 'res_c' in st.session_state:
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("チュナ消費量", f"{int(st.session_state['res_c']):,}")
        m2.metric("レコード消費量", f"{int(st.session_state['res_r']):,}")
        m3.metric("音骸素体消費量", f"{st.session_state['res_b']:.1f} 個" if st.session_state['res_b'] < 10000 else "∞")

# --- Tab 2: 続行判定 ---
with tab2:
    st.header("強化続行・撤退の判定")
    st.markdown(f"目標スコア **{int(st.session_state['target_score'])}** を目指す場合の判定を行います。")
    
    col_in, col_re = st.columns([1, 1])
    with col_in:
        st.subheader("現在のステータス")
        times = st.slider("現在のサブステ開放数", 0, 4, 1)
        # ヘルパー関数でスライダー生成
        sub = render_substat_inputs(current_sub_names, subst_list, coe, "tab2")
        
    with col_re:
        st.subheader("判定結果")
        sc_now = cal_score_now(sub, coe)
        st.metric("現在のスコア", f"{sc_now:.2f}")
        
        if st.button("判定実行", type="primary"):
            cst, msg = judge_continue(st.session_state['target_score'], times, sub, st.session_state['ave_chuna'], coe)
            if msg == "強化推奨": st.success(f"## {msg}")
            elif "続行可能" in msg: st.warning(f"## {msg}")
            else: st.error(f"## {msg}")
            
            if cst < INF: st.write(f"ゴールまでの期待コスト: **{int(cst)}** チュナ")

# --- Tab 3: 最小ライン一覧 ---
with tab3:
    st.header("これ以上強化していい最小ライン一覧")
    search_t = st.selectbox("検索する強化回数", [1, 2, 3, 4])
    
    if st.button("一覧を生成"):
        with st.spinner("探索中..."):
            res = judge_continue_all(st.session_state['target_score'], search_t, st.session_state['ave_chuna'], coe)
            
            if not res:
                st.warning("条件を満たす組み合わせがありません")
            else:
                rows = []
                for r in res:
                    row = {current_sub_names[i]: r["substatus"][i] for i in range(7)}
                    row.update({"スコア": r["score"], "消費チュナ": r["chuna"]})
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                # 表示カラムの整理（係数>0のみ）
                act_cols = [n for i, n in enumerate(current_sub_names) if coe[i] > 0]
                df_disp = df[act_cols + ["スコア", "消費チュナ"]].sort_values("スコア")
                
                # 数値フォーマット定義
                fmt = {c: st.column_config.NumberColumn(format="%.1f") for c in act_cols+["スコア"]}
                fmt["消費チュナ"] = st.column_config.NumberColumn(format="%d")
                
                st.dataframe(
                    df_disp.style.map(style_zeros),
                    column_config=fmt,
                    use_container_width=True,
                    hide_index=True
                )

# --- Tab 4: スコア計算(単体) ---
with tab4:
    st.subheader("① 音骸スコア計算（単体）")
    
    # ヘルパー関数で入力欄生成
    sub_s = render_substat_inputs(current_sub_names, subst_list, coe, "tab4")
    
    st.divider()
    if sum(1 for v in sub_s if v > 0) > 5:
        st.error("有効サブステが多すぎます")
    else:
        sc_s = cal_score_now(sub_s, coe)
        st.metric("合計スコア", f"{sc_s:.2f}")
        
        # 内訳表
        df_s = pd.DataFrame({
            "サブステ": current_sub_names,
            "値": sub_s,
            "スコア寄与": [sub_s[i]*coe[i] for i in range(7)]
        })
        st.dataframe(df_s[df_s["値"]>0], use_container_width=True, hide_index=True)
        
        # CSV DL
        csv = df_s.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 結果をCSVでダウンロード", csv, "score.csv", "text/csv")

# --- Tab 5: 5連音骸管理 ---
with tab5:
    st.subheader("② 5連音骸スコア管理")
    echo_labels = ["コスト4", "コスト3(A)", "コスト3(B)", "コスト1(A)", "コスト1(B)"]
    
    # データ保持用辞書
    all_stats_dict = {name: [0.0] * 5 for name in current_sub_names if coe[current_sub_names.index(name)] > 0}
    
    # 5連入力フォーム
    for echo_idx, label in enumerate(echo_labels):
        with st.expander(f"{label} の入力", expanded=(echo_idx==0)):
            # ヘルパー関数で各部位の入力欄を生成
            sub_inputs = render





    




