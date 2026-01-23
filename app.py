import streamlit as st
import numpy as np
import pandas as pd
import copy
import itertools

#チュナ最小になる戦略（新）(レコードの消費量も計算)

coe=[2,1,1,0,0,0,0.1]

cr_list=[6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5]
cd_list=[12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0]
at_p_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage1_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage2_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
at_n_list=[30,40,50,60,0,0,0,0]
efficiency_list=[6.8,7.6,8.4,9.2,10,10.8,11.6,12.4]

cr_weight=[0.2429, 0.2458, 0.2023, 0.0940, 0.0840, 0.0872, 0.0269, 0.0169]
cd_weight=[0.1575091575091575, 0.27472527472527475, 0.28205128205128205, 0.07692307692307693, 0.09157509157509157, 0.08058608058608059, 0.02197802197802198,0.014652014652014652]
at_p_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
damage1_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
damage2_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
efficiency_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
at_n_weight=[0.269,0.385,0.325,0.021,0,0,0,0]

subst_list=[cr_list,cd_list,at_p_list,damage1_list,damage2_list,efficiency_list,at_n_list]
subst_weight=[cr_weight,cd_weight,at_p_weight,damage1_weight,damage2_weight,efficiency_weight,at_n_weight]
record_list=[1100,3025,5775,9875,15875]

def cal_score_now(substatus):#現在スコア
    s=0
    for i in range(7):
        s+=substatus[i]*coe[i]
    return s


def number_effective_subst(coe):
    n=0
    for i in range(7):
        if coe[i]>0:
            n+=1
    return n       

def cal_ave_chuna4(score,substatus):
    chuna=0
    record=0
    prob=0
    score_now=cal_score_now(substatus)
    if score_now>=score:
        return 10,1,record_list[4]*4
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for k in range(8):
            if coe[i]*subst_list[i][k]+score_now>=score:
                prob+=subst_weight[i][k]
    prob=prob/9
    if prob==0:
        return chuna,prob,record
    chuna=7/prob+3
    record=record_list[4]/prob+record_list[4]*3
    return chuna,prob,record

def cal_effective_subst(substatus):#有効サブステの数
    n=0
    for i in range(7):
        if substatus[i]>0:
            n+=1
    return n           
            

def cal_ave_chuna3(score,substatus,ave_chuna):
    chuna=0
    record=0
    prob1=0 #1回だけ強化
    prob2=0 #2回とも強化して成功
    score_now=cal_score_now(substatus)
    
    if score_now>=score:
        record=(record_list[3]+record_list[4])*4
        return 20,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna4=cal_ave_chuna4(score,substatus_next)
            if cal_chuna4[1]>0 and cal_chuna4[0]<=ave_chuna:
                prob2+=subst_weight[i][j]*cal_chuna4[1]
            else:
                prob1+=subst_weight[i][j]
    
    prob1=prob1/10
    prob2=prob2/10
    cal_chuna4=cal_ave_chuna4(score,substatus)
    
    if cal_chuna4[1]>0 and cal_chuna4[0]<=ave_chuna:
        prob2+=cal_chuna4[1]*(10-number_effective_subst(coe)+cal_effective_subst(substatus))/10
    else:
        prob1+=(10-number_effective_subst(coe)+cal_effective_subst(substatus))/10
        
    if prob2==0:
        return chuna,prob2,record
    
    chuna=(14-7*prob1)/prob2+6
    record=(record_list[4]+record_list[3]-record_list[4]*prob1)/prob2+(record_list[3]+record_list[4])*3
    return chuna,prob2,record      
                
def cal_ave_chuna2(score,substatus,ave_chuna):
    chuna=0
    record=0
    chuna_1time_1=0 #3回目に有効ステが着いたときの消費チュナ期待値
    chuna_1time_2=0 #3回目に不要ステが着いたときの消費チュナ期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #3回とも強化して成功
    score_now=cal_score_now(substatus)
    
    if score_now>=score:
        record=(record_list[2]+record_list[3]+record_list[4])*4
        return 30,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna3=cal_ave_chuna3(score,substatus_next,ave_chuna)
            if cal_chuna3[1]>0 and cal_chuna3[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna3[0]-6)*cal_chuna3[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna3[2]-(record_list[3]+record_list[4])*3)*cal_chuna3[1]
                prob+=subst_weight[i][j]*cal_chuna3[1]
    
    cal_chuna3=cal_ave_chuna3(score,substatus,ave_chuna)
    
    if cal_chuna3[1]>0 and cal_chuna3[0]<=ave_chuna:
        x=cal_chuna3[1]*(11-number_effective_subst(coe)+cal_effective_subst(substatus))
        prob+=x
        chuna_1time_2+=x*(cal_chuna3[0]-6)
        record_1time_2+=x*(cal_chuna3[2]-(record_list[3]+record_list[4])*3)
        
    if prob==0:
        return chuna,prob,record
    
    chuna=(chuna_1time_1+chuna_1time_2+7*11)/prob+9
    record=(record_1time_1+record_1time_2+record_list[2]*11)/prob+(record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/11,record 

def cal_ave_chuna1(score,substatus,ave_chuna):
    chuna=0
    record=0
    chuna_1time_1=0 #2回目に有効ステが着いたときの消費チュナ期待値
    chuna_1time_2=0 #2回目に不要ステが着いたときの消費チュナ期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #4回とも強化して成功
    score_now=cal_score_now(substatus)
    
    if score_now>=score:
        record=(record_list[1]+record_list[2]+record_list[3]+record_list[4])*4
        return 40,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna2=cal_ave_chuna2(score,substatus_next,ave_chuna)
            if cal_chuna2[1]>0 and cal_chuna2[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna2[0]-9)*cal_chuna2[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna2[2]-(record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna2[1]
                prob+=subst_weight[i][j]*cal_chuna2[1]

    cal_chuna2=cal_ave_chuna2(score,substatus,ave_chuna)
    
    if cal_chuna2[1]>0 and cal_chuna2[0]<=ave_chuna:
        x=cal_chuna2[1]*(12-number_effective_subst(coe)+cal_effective_subst(substatus))
        prob+=x
        chuna_1time_2+=x*(cal_chuna2[0]-9)
        record_1time_2+=x*(cal_chuna2[2]-(record_list[2]+record_list[3]+record_list[4])*3)
        
    if prob==0:
        return chuna,prob,record
    
    chuna=(chuna_1time_1+chuna_1time_2+7*12)/prob+12
    record=(record_1time_1+record_1time_2+record_list[1]*12)/prob+(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/12,record  

def cal_ave_chuna0(score,ave_chuna):
    substatus=[0,0,0,0,0,0,0]
    chuna_1time_1=0 #1回目に有効ステが着いて強化継続したときのチュナ消費期待値
    chuna_1time_2=0 #1回目に不要ステが着いて強化継続したときのチュナ消費期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #最後まで強化して成功
    for i in range(7):
        if coe[i]==0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna1=cal_ave_chuna1(score,substatus_next,ave_chuna)
            if cal_chuna1[1]>0 and cal_chuna1[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna1[0]-12)*cal_chuna1[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna1[2]-(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna1[1]
                prob+=subst_weight[i][j]*cal_chuna1[1]
                
    cal_chuna1=cal_ave_chuna2(score,substatus,ave_chuna)
    if cal_chuna1[1]>0 and cal_chuna1[0]<=ave_chuna:
        chuna_1time_2+=(13-number_effective_subst(coe))*(cal_chuna1[0]-12)*cal_chuna1[1]
        record_1time_2+=(13-number_effective_subst(coe))*(cal_chuna1[2]-(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna1[1]
        prob+=(13-number_effective_subst(coe))*cal_chuna1[1]
    if prob==0:
        return 0,0,0
    chuna=(chuna_1time_1+chuna_1time_2+7*13)/prob+15
    record=(record_1time_1+record_1time_2+record_list[0]*13)/prob+(record_list[0]+record_list[1]+record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/13,record

def cal_min_chuna(score):#入力スコア以上の音骸を一つ作るのに必要なチュナ
    chuna1=4000
    chuna2=0
    while abs(chuna1-chuna2)>=1:
        chuna2=chuna1
        chuna1=cal_ave_chuna0(score,chuna1)[0]
    return chuna1    

def judge_continue(score,times,substatus,ave_chuna):#強化続行判定
    chuna=0
    if times==1:
        chuna=cal_ave_chuna1(score,substatus,ave_chuna)[0]
    elif times==2:
        chuna=cal_ave_chuna2(score,substatus,ave_chuna)[0]
    elif times==3:
        chuna=cal_ave_chuna3(score,substatus,ave_chuna)[0]
    elif times==4:
        chuna=cal_ave_chuna4(score,substatus)[0]
    else:
        return -1,"error"
    
    if chuna<=ave_chuna:
        return int(chuna),"強化推奨"
    elif chuna<=ave_chuna*1.2:
        return int(chuna),"続行可能"
    else:
        return int(chuna),"強化非推奨"
   
def cal_ave_chuna(score,times,substatus,ave_chuna):
    if times==0:
        return cal_ave_chuna0(score,ave_chuna)
    elif times==1:
        return cal_ave_chuna1(score,substatus,ave_chuna)
    elif times==2:
        return cal_ave_chuna2(score,substatus,ave_chuna)
    elif times==3:
        return cal_ave_chuna3(score,substatus,ave_chuna)
    elif times==4:
        return cal_ave_chuna4(score,substatus)
    else:
        print("error")

def is_dominated(vec, memory):
    """
    有効サブステ（coe > 0）について、
    すでに列挙済みのものよりすべて劣っているなら除外
    """
    for m in memory:
        dominated = True
        for i in range(7):
            if coe[i] == 0:
                continue
            if m[i] > vec[i]:
                dominated = False
                break
        if dominated:
            return True
    return False

def enumerate_valid_substats(score, times, ave_chuna):
    results = []
    memory = []

    subst_list0 = copy.deepcopy(subst_list)
    for i in range(7):
        subst_list0[i].insert(0, 0)

    for comb in itertools.combinations(range(7), times):
        ranges = []
        for i in comb:
            if i == 6:  # 攻撃力実数
                ranges.append(range(1, 5))
            else:
                ranges.append(range(1, 9))

        for levels in itertools.product(*ranges):
            vec = np.zeros(7, dtype=int)
            for i, lv in zip(comb, levels):
                vec[i] = lv

            if is_dominated(vec, memory):
                continue

            substatus = [subst_list0[i][vec[i]] for i in range(7)]
            chuna = cal_ave_chuna(score, times, substatus, ave_chuna)

            if chuna[1] == 0:
                continue

            if chuna[0] <= ave_chuna:
                memory.append(vec)
                results.append({
                    "クリ率": substatus[0],
                    "クリダメ": substatus[1],
                    "攻撃%": substatus[2],
                    "ダメUP1": substatus[3],
                    "ダメUP2": substatus[4],
                    "共鳴効率": substatus[5],
                    "攻撃実数": substatus[6],
                    "有効サブ数": sum(1 for i in range(7) if substatus[i] > 0 and coe[i] > 0),
                    "合計スコア": round(cal_score_now(substatus), 2),
                    "期待チュナ": round(chuna[0], 2),
                })

    return results


       
# =========================
# タイトル
# =========================
st.title("音骸厳選用計算ツール")

# =========================
# キャッシュ付き計算
# =========================
@st.cache_data
def cached_cal_min_chuna(score):
    return cal_min_chuna(score)

# =========================
# 共通パラメータ入力
# =========================
st.header("基本設定")
st.write("スコア計算に必要な各サブステータスの係数を入力")
st.caption("一般的なアタッカーは○○ダメージアップの係数は0.7に設定すれば、それほど支障はない")
st.caption("共鳴効率はサブステータスで30程度盛りたい場合は1、20程度盛りたい場合は0.6と設定することを推奨")
st.caption("消滅漂泊者など2種類のダメアップが有効なキャラ以外は、○○ダメージアップ2のところは0のままでOK")
st.caption("現在、HP参照キャラであるカルテジアは非対応")

coe = [2, 1, 1, 0, 0, 0, 0.1]

coe[3] = st.number_input("○○ダメージアップ1", value=0.0, step=0.1)
coe[4] = st.number_input("○○ダメージアップ2", value=0.0, step=0.1)
coe[5] = st.number_input("共鳴効率", value=0.0, step=0.1)

score = st.number_input("目標スコア", min_value=1, step=1)

# =========================
# ① 必要素材計算
# =========================
st.header("① 目標スコア以上の音骸を1体作るための素材消費量")

if st.button("① 計算する"):
    with st.spinner("計算中..."):
        chuna = cached_cal_min_chuna(score)
        n = list(cal_ave_chuna0(score, chuna))

    n[0] = int(n[0])
    n[2] = int(n[2] / 5000)

    if n[1] > 0:
        n.append(int(1 / n[1]))
    else:
        n.append("無限大")

    # セッションに保存（②で使う）
    st.session_state["ave_chuna"] = n[0]

    st.subheader("計算結果")
    st.metric("チュナ消費量", n[0])
    st.metric("レコード消費量", n[2])
    st.metric("素体消費量", n[3])

# =========================
# ② 強化続行判定
# =========================
st.header("② 音骸の強化続行判定")

st.caption("①の計算結果を元に判定します")

times = st.number_input(
    "強化回数（サブステが開いた数）",
    min_value=0,
    max_value=4,
    step=1
)

substatus = [0.0] * 7
substatus[0] = st.number_input("クリティカル", value=0.0)
substatus[1] = st.number_input("クリティカルダメージ", value=0.0)
substatus[2] = st.number_input("攻撃力％", value=0.0)
substatus[3] = st.number_input("○○ダメージアップ1", value=0.0)
substatus[4] = st.number_input("○○ダメージアップ2", value=0.0)
substatus[5] = st.number_input("共鳴効率", value=0.0)
substatus[6] = st.number_input("攻撃力実数", value=0.0)

if st.button("② 判定する"):
    if "ave_chuna" not in st.session_state:
        st.error("先に①の計算を実行してください")
        st.stop()

    ave_chuna = st.session_state["ave_chuna"]

    with st.spinner("判定中..."):
        result = list(judge_continue(score, times, substatus, ave_chuna))

    st.subheader("判定結果")
    st.metric("想定チュナ消費量", result[0])
    st.write(result[1])
    
st.header("③強化続行サブステ一覧")
st.caption("①の計算結果をもとに表示します")

times = st.number_input(
    "強化回数（開放サブステ数）",
    min_value=0,
    max_value=5,
    step=1
)

if st.button("一覧を表示"):
    if "ave_chuna" not in st.session_state:
        st.error("先に①の計算を実行してください")
        st.stop()

    ave_chuna = st.session_state["ave_chuna"]

    with st.spinner("計算中…（少し時間がかかります）"):
        data = enumerate_valid_substats(score, times, ave_chuna)

    if len(data) == 0:
        st.warning("条件を満たすサブステ構成がありません")
    else:
        df = pd.DataFrame(data)

        st.subheader(f"続行推奨サブステ一覧（{len(df)} 件）")

        st.dataframe(
            df.sort_values("期待チュナ"),
            use_container_width=True,
            height=600
        )







    




