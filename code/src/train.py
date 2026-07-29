"""
XGBRanker 排序学习训练脚本
标签：超额收益（已由 _build_label_and_clean 计算）
特征：将 60 天序列展平为单行特征向量（60 × 197 = 11,820 维）
分组：每个交易日为一个 group（qid），group 内股票按超额收益排序
"""

import os
import json
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import config, config_extended, xgb_config
from utils import engineer_features_39, engineer_features_158plus39
from evaluation import calculate_extended_metrics, format_eval_report


# ============================================================
#  特征列映射 & 特征工程（复用 utils.py，不修改特征逻辑）
# ============================================================

feature_columns_map = {
    '39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
#  标签构建：超额收益（不改计算逻辑）
# ============================================================

def _build_label_and_clean(processed, drop_small_open=True):
    """构建超额收益标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)

    # 转换为超额收益：减去当日等权指数收益
    processed['_daily_mean'] = processed.groupby('日期')['label'].transform('mean')
    processed['label'] = processed['label'] - processed['_daily_mean']
    processed.drop(columns=['_daily_mean'], inplace=True)

    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


# ============================================================
#  数据预处理（复用 utils.py 特征工程）
# ============================================================

def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


# ============================================================
#  基本面因子合并
# ============================================================

FUNDAMENTAL_COLS = ['PE_TTM', 'PB', 'ROE_approx', '总市值_对数']

# ── 行业聚类映射：430+ 细分行业 → 15 大类（申万一级分类）──
_INDUSTRY_BROAD_MAP = {
    # 金融
    '银行': '金融', '国有银行': '金融', '国有大型银行Ⅲ': '金融', '国有大型银': '金融',
    '股份制银行': '金融', '股份制银行Ⅲ': '金融', '城商行Ⅲ': '金融', '农商行Ⅲ': '金融',
    '区域性银行': '金融', '综合性银行': '金融', '其他商业银行': '金融',
    '保险': '金融', '保险业': '金融', '财产保险': '金融', '人身保险': '金融',
    '人寿与健康保险': '金融', '多元化保险': '金融', '财产与意外伤害保险': '金融',
    '证券': '金融', '证券公司': '金融', '证券Ⅲ': '金融', '证券及经纪': '金融',
    '投资银行业与经纪业': '金融', '资本市场服务': '金融', '金融控股': '金融',
    '其他金融业': '金融', '其他金融服务': '金融', '其他综合性金融服务': '金融',
    '多元金融': '金融', '货币金融服务': '金融', '综合性金融服务': '金融',
    '金融交易与数据': '金融', '金融数据服务': '金融', '投资及资产管理': '金融',
    # 食品饮料
    '白酒': '食品饮料', '白酒Ⅲ': '食品饮料', '啤酒': '食品饮料', '乳制品': '食品饮料',
    '乳品': '食品饮料', '调味品': '食品饮料', '调味发酵品Ⅲ': '食品饮料',
    '调味品与食品添加剂': '食品饮料', '调味品与食用油': '食品饮料',
    '软饮料': '食品饮料', '肉制品': '食品饮料', '包装食品与肉类': '食品饮料',
    '食品饮料': '食品饮料', '食品制造业': '食品饮料', '食品及饲料添加剂': '食品饮料',
    '酒精饮料': '食品饮料', '酒、饮料和精制茶制造业': '食品饮料',
    # 医药生物
    '医药': '医药生物', '医药生物': '医药生物', '医药制造业': '医药生物',
    '化学药': '医药生物', '化学制剂': '医药生物', '化学原料药': '医药生物', '原料药': '医药生物',
    '中药': '医药生物', '中成药': '医药生物', '生物制品': '医药生物',
    '生物技术': '医药生物', '生物科技': '医药生物', '疫苗': '医药生物',
    '医疗器械': '医药生物', '医疗设备': '医药生物', '医疗用品': '医药生物',
    '医疗耗材': '医药生物', '体外诊断': '医药生物', '医美耗材': '医药生物',
    '医疗服务': '医药生物', '医疗保健': '医药生物', '医疗保健机构': '医药生物',
    '医疗保健机构与服务': '医药生物', '医疗保健设备': '医药生物',
    '医疗研发外包': '医药生物', '医院': '医药生物', '卫生': '医药生物',
    '医药商业': '医药生物', '医药流通': '医药生物', '药品流通': '医药生物',
    '药品经销商': '医药生物', '药品零售': '医药生物', '药品': '医药生物',
    '药品制剂': '医药生物', '药品及医疗器械批发业': '医药生物',
    '制药与生物科技服务': '医药生物', '生命科学工具和服务': '医药生物',
    '研究和试验发展': '医药生物',
    # 电子
    '电子': '电子', '半导体': '电子', '集成电路': '电子', '集成电路设计': '电子',
    '集成电路制造': '电子', '集成电路封测': '电子', '数字芯片设计': '电子',
    '模拟芯片设计': '电子', '消费电子': '电子', '消费电子产品': '电子',
    '消费电子终端': '电子', '消费电子零部件及组装': '电子', '消费电子组件及零部件': '电子',
    '品牌消费电子': '电子', '面板': '电子', '光电子器件': '电子',
    '被动元件': '电子', '印制电路板': '电子', '分立器件': '电子',
    '电子零部件制造': '电子', '电子元器件': '电子', '电子制造服务': '电子',
    '电子设备与仪器': '电子', '电子设备及仪表制造商': '电子', '电子系统组装': '电子',
    '其他电子': '电子', '安防设备': '电子', '安防设备及其他': '电子',
    'LED': '电子', '视听器材': '电子',
    # 计算机
    '计算机': '计算机', 'IT服务': '计算机', '软件开发': '计算机',
    '行业应用软件': '计算机', '应用软件': '计算机', '横向通用软件': '计算机',
    '垂直应用软件': '计算机', '通用软件': '计算机', '系统集成及IT咨询': '计算机',
    '云计算服务': '计算机', '互联网': '计算机', '互联网和相关服务': '计算机',
    '互联网信息服务': '计算机', '互联网软件与服务': '计算机', '互动媒体': '计算机',
    '移动互联网': '计算机', '移动互联网信息服务': '计算机', '移动互联网服务': '计算机',
    '信息科技咨询与其他服务': '计算机', '系统开发及资讯科技顾问': '计算机',
    '游戏': '计算机', '游戏Ⅲ': '计算机', '视频媒体': '计算机', '广告': '计算机',
    '广告媒体': '计算机', '网络零售': '计算机', '数据中心': '计算机',
    '软件和信息技术服务业': '计算机', '电脑硬件': '计算机', '电脑存储与外围设备': '计算机',
    '电脑及电子设备经销商': '计算机', '电脑与外围设备': '计算机',
    '计算机、通信和其他电子设备制造业': '计算机', '其他IT与互联网服务': '计算机',
    '营销与广告': '计算机', '营销服务': '计算机', '其他广告营销': '计算机',
    '其他计算机设备': '计算机',
    # 电力设备/新能源
    '电力设备': '电力新能源', '电池': '电力新能源', '锂电池': '电力新能源',
    '电池化学品': '电力新能源', '电池部件及材料': '电力新能源',
    '光伏': '电力新能源', '光伏设备': '电力新能源', '光伏电池组件': '电力新能源',
    '光伏加工设备': '电力新能源', '逆变器': '电力新能源',
    '风电': '电力新能源', '风力发电': '电力新能源', '风电整机': '电力新能源',
    '风电设备': '电力新能源', '新能源': '电力新能源', '新能源发电': '电力新能源',
    '新能源设备': '电力新能源', '储能设备': '电力新能源', '其他储能设备': '电力新能源',
    '核电': '电力新能源', '核力发电': '电力新能源', '能源设备': '电力新能源',
    '硅料硅片': '电力新能源', '电网自动化': '电力新能源', '电网自动化设备': '电力新能源',
    '输变电设备': '电力新能源', '配电设备': '电力新能源', '高压设备': '电力新能源',
    '低压设备': '电力新能源', '电气机械和器材制造业': '电力新能源',
    '电气部件与设备': '电力新能源', '电动机与工控自动化': '电力新能源',
    '工控自动化': '电力新能源', '工控设备': '电力新能源', '激光设备': '电力新能源',
    '电源设备': '电力新能源', '发电设备': '电力新能源', '其他电源设备Ⅲ': '电力新能源',
    '其他发电设备': '电力新能源', '重型电气设备': '电力新能源',
    # 汽车
    '汽车': '汽车', '乘用车': '汽车', '轿车': '汽车', '商用车': '汽车',
    '商用载客车': '汽车', '综合乘用车': '汽车', '电动乘用车': '汽车',
    '汽车零部件': '汽车', '汽车零件': '汽车', '汽车零配件': '汽车',
    '汽车制造业': '汽车', '汽车和汽车零部件': '汽车',
    '汽车电子': '汽车', '汽车系统部件': '汽车', '汽车内饰与外饰': '汽车',
    '车身附件及饰件': '汽车', '轮胎': '汽车', '轮胎轮毂': '汽车',
    '底盘与发动机系统': '汽车', '发动机与涡轮机': '汽车',
    '商业用车及货车': '汽车',
    # 家用电器
    '家电': '家用电器', '家用电器': '家用电器', '白色家电': '家用电器',
    '家庭电器': '家用电器', '空调': '家用电器', '冰箱': '家用电器',
    '冰洗': '家用电器', '家电零部件': '家用电器', '家电零部件Ⅲ': '家用电器',
    '家电零部件及其他': '家用电器', '视听器材': '家用电器',
    # 机械设备
    '机械': '机械设备', '通用机械': '机械设备', '通用设备制造业': '机械设备',
    '其他通用机械': '机械设备', '专用设备': '机械设备', '其他专用设备': '机械设备',
    '其他专用机械': '机械设备', '专用设备制造业': '机械设备', '其它专用机械': '机械设备',
    '工程机械': '机械设备', '工程机械整机': '机械设备', '工程机械器件': '机械设备',
    '重机械': '机械设备', '重型基建': '机械设备', '其他机械设备': '机械设备',
    '锂电专用设备': '机械设备', '冶金矿采化工设备': '机械设备',
    '制冷设备': '机械设备', '气液机械': '机械设备', '楼宇设备': '机械设备',
    '印刷包装机械': '机械设备', '建筑工程与运输机械': '机械设备',
    '工业地产开发和管理': '机械设备',
    # 国防军工
    '军工': '国防军工', '国防': '国防军工', '国防装备': '国防军工',
    '军工电子Ⅲ': '国防军工', '航天': '国防军工', '航天装备': '国防军工',
    '航天航空': '国防军工', '航空航天与国防': '国防军工', '航空装备': '国防军工',
    '航空装备Ⅲ': '国防军工', '地面兵装': '国防军工', '航海装备Ⅲ': '国防军工',
    '船舶制造': '国防军工', '船舶及其他航运设备': '国防军工',
    '铁路、船舶、航空航天和其他运输设备制造业': '国防军工',
    # 有色金属/钢铁
    '有色金属': '有色钢铁', '基本金属': '有色钢铁', '稀有金属': '有色钢铁',
    '其他稀有金属': '有色钢铁', '贵金属': '有色钢铁', '黄金': '有色钢铁',
    '黄金及其它贵金属': '有色钢铁', '黄金及贵金属': '有色钢铁',
    '铜': '有色钢铁', '铝': '有色钢铁', '铅锌': '有色钢铁', '钨': '有色钢铁',
    '钨钼': '有色钢铁', '钼': '有色钢铁', '钴': '有色钢铁', '钴镍': '有色钢铁',
    '锂': '有色钢铁', '稀土': '有色钢铁', '稀土金属': '有色钢铁',
    '有色金属冶炼和压延加工业': '有色钢铁', '有色金属矿采选业': '有色钢铁',
    '其它有色金属及合金': '有色钢铁',
    '钢铁': '有色钢铁', '普钢': '有色钢铁', '特钢': '有色钢铁', '特钢Ⅲ': '有色钢铁',
    '板材': '有色钢铁', '黑色金属冶炼和压延加工业': '有色钢铁',
    '金属制品业': '有色钢铁', '合成金属': '有色钢铁',
    '其他金属及矿物': '有色钢铁', '非金属采矿及制品': '有色钢铁',
    # 化工
    '化工': '化工', '基础化工': '化工', '化学制品': '化工',
    '其他化学制品': '化工', '化学原料': '化工', '其他化学原料': '化工',
    '化学原料和化学制品制造业': '化工', '化学纤维制造业': '化工',
    '化学工程': '化工', '煤化工': '化工', '石油化工': '化工', '石化': '化工',
    '其他石化': '化工', '炼油化工': '化工', '燃油炼制': '化工',
    '氟化工': '化工', '氟化工及制冷剂': '化工', '有机硅': '化工',
    '聚氨酯': '化工', '化肥': '化工', '化肥与农药': '化工',
    '氮肥': '化工', '钾肥': '化工', '涤纶': '化工', '锦纶与涤纶': '化工',
    '合成纤维': '化工', '纤维及树脂': '化工', '橡胶和塑料制品业': '化工',
    # 房地产/建筑
    '房地产': '地产建筑', '房地产开发': '地产建筑', '房地产业': '地产建筑',
    '住宅开发': '地产建筑', '住宅房地产开发': '地产建筑', '住宅地产开发和管理': '地产建筑',
    '商业地产': '地产建筑', '商业地产开发和管理': '地产建筑',
    '商业物业经营': '地产建筑', '房地产租赁': '地产建筑',
    '地产发展商': '地产建筑', '工业地产开发和管理': '地产建筑',
    '园区': '地产建筑', '房屋建设': '地产建筑', '房屋建设Ⅲ': '地产建筑',
    '建筑': '地产建筑', '建筑施工': '地产建筑', '建筑工程': '地产建筑',
    '基础设施建设': '地产建筑', '基建市政工程': '地产建筑', '土木工程': '地产建筑',
    '土木工程建筑业': '地产建筑', '路桥施工': '地产建筑', '水利工程': '地产建筑',
    '建筑材料': '地产建筑', '水泥': '地产建筑', '水泥与混凝土': '地产建筑',
    '玻璃纤维': '地产建筑', '玻纤': '地产建筑', '玻纤制造': '地产建筑',
    '非金属材料': '地产建筑', '非金属新材料': '地产建筑',
    '非金属材料Ⅲ': '地产建筑', '非金属材料与制品': '地产建筑',
    '非金属矿物制品业': '地产建筑', '其他非金属材料': '地产建筑',
    # 交通运输
    '交通运输': '交通运输', '公路与铁路': '交通运输', '高速公路': '交通运输',
    '铁路运输': '交通运输', '铁路运输业': '交通运输', '铁路设备': '交通运输',
    '城轨铁路': '交通运输', '航空运输': '交通运输', '航空运输业': '交通运输',
    '航空': '交通运输', '航空公司': '交通运输', '航空服务': '交通运输',
    '航运': '交通运输', '航运及港口': '交通运输', '港口': '交通运输',
    '港口服务': '交通运输', '水上运输': '交通运输', '水上运输业': '交通运输',
    '物流': '交通运输', '快递': '交通运输', '跨境物流': '交通运输',
    '交通运输仓储': '交通运输', '道路运输业': '交通运输', '邮政业': '交通运输',
    '机场': '交通运输', '机场服务': '交通运输',
    # 公用事业
    '电力': '公用事业', '电力公用事业': '公用事业', '电力、热力生产和供应业': '公用事业',
    '电力、煤气及水等公用事业': '公用事业', '火电': '公用事业', '水电': '公用事业',
    '热电': '公用事业', '燃气': '公用事业', '燃气公用事业': '公用事业',
    '燃气生产和供应业': '公用事业',
    # 能源/煤炭
    '煤炭': '能源', '煤炭开采': '能源', '煤炭开采和洗选业': '能源',
    '动力煤': '能源', '焦炭': '能源', '石油': '能源',
    '石油和天然气开采业': '能源', '油气开采': '能源',
    '石油与天然气开采设备与服务': '能源', '气油生产商': '能源',
    '综合性石油与天然气企业': '能源', '综合性石油天然气企业': '能源',
    # 通信
    '通信': '通信', '通信设备': '通信', '通信系统设备及组件': '通信',
    '通信网络设备及器件': '通信', '通信传输设备': '通信', '通信终端设备': '通信',
    '通信终端及配件': '通信', '通信终端设备及组件': '通信',
    '通信线缆及配套': '通信', '通信技术服务': '通信',
    '通信应用增值服务': '通信', '电讯设备': '通信',
    '电信': '通信', '电信运营': '通信', '电信运营商': '通信', '电信运营服务': '通信',
    '电信增值服务': '通信', '电信、广播电视和卫星传输服务': '通信',
    '广播、电视、电影和录音制作业': '通信', '广播、电视、电影和影视录音制作业': '通信',
    '终端设备': '通信',
    # 传媒
    '传媒': '传媒', '电影与娱乐': '传媒', '影视动漫': '传媒',
    '影视动漫制作': '传媒', '文化艺术业': '传媒',
    # 农林牧渔
    '农林牧渔': '农林牧渔', '农业': '农林牧渔', '畜牧业': '农林牧渔',
    '畜牧产品': '农林牧渔', '饲料': '农林牧渔', '水产饲料': '农林牧渔',
    '农产品': '农林牧渔', '其他农产品': '农林牧渔', '生猪养殖': '农林牧渔',
    '农副食品加工业': '农林牧渔',
    # 消费/零售
    '零售': '消费零售', '零售业': '消费零售', '旅游零售': '消费零售',
    '旅游零售Ⅲ': '消费零售', '市场服务': '消费零售', '商务服务业': '消费零售',
    '旅游': '消费零售', '旅游及观光': '消费零售', '旅行社': '消费零售',
    # 综合/其他
    '综合Ⅲ': '综合', '环保': '综合', '环保工程': '综合', '环保工程及服务': '综合',
    '专业工程': '综合', '专业市场': '综合', '其他专业工程': '综合',
    '未知': '未知',
}


def _map_industry_broad(industry_name):
    """将细分行业名映射到大类。先精确匹配，再模糊匹配。"""
    if not isinstance(industry_name, str):
        return '未知'
    # 精确匹配
    if industry_name in _INDUSTRY_BROAD_MAP:
        return _INDUSTRY_BROAD_MAP[industry_name]
    # 模糊匹配：检查行业名是否包含已知的键
    for key, broad in _INDUSTRY_BROAD_MAP.items():
        if key in industry_name:
            return broad
    return '其他'


def _merge_fundamentals(processed, fundamental_path):
    """
    将基本面因子（PE/PB/ROE/市值）按日期+股票代码合并到特征 DataFrame。
    支持时序型基本面文件（含'日期'列）和静态快照（无'日期'列）两种格式。
    """
    if not os.path.exists(fundamental_path):
        print(f"  ⚠ 基本面数据不存在: {fundamental_path}，跳过合并")
        return processed, []

    fund_df = pd.read_csv(fundamental_path, dtype={'股票代码': str})
    fund_df['股票代码'] = fund_df['股票代码'].astype(str).str.zfill(6)
    # 确保 processed 的股票代码也是字符串（cross_val 切片可能产生 int64）
    processed['股票代码'] = processed['股票代码'].astype(str).str.zfill(6)
    # 确保日期列存在且格式统一
    if '日期' in processed.columns:
        processed['_date_merge'] = pd.to_datetime(processed['日期']).dt.strftime('%Y-%m-%d')
    if '日期' in fund_df.columns:
        fund_df['_date_merge'] = pd.to_datetime(fund_df['日期'], errors='coerce').dt.strftime('%Y-%m-%d')

    available_cols = [c for c in FUNDAMENTAL_COLS if c in fund_df.columns]
    if not available_cols:
        return processed, []

    # 行业 one-hot 编码（先映射到大类，避免 430 维过拟合）
    if '行业' in fund_df.columns:
        fund_df['行业_大类'] = fund_df['行业'].fillna('未知').apply(_map_industry_broad)
        industry_dummies = pd.get_dummies(fund_df['行业_大类'], prefix='行业')
        fund_df = fund_df.join(industry_dummies)
        available_cols += list(industry_dummies.columns)
        broad_count = industry_dummies.shape[1]
        print(f"  行业聚类: {fund_df['行业'].nunique()} 细分 → {broad_count} 大类")

    # 合并：如果两边都有日期则按时序合并，否则按静态快照合并
    merge_keys = ['股票代码']
    if '_date_merge' in processed.columns and '_date_merge' in fund_df.columns:
        merge_keys.append('_date_merge')
        print(f"  基本面合并模式: 时序对齐 (按股票+日期)")
    else:
        print(f"  基本面合并模式: 静态快照 (按股票代码)")

    processed = processed.merge(
        fund_df[merge_keys + available_cols],
        on=merge_keys, how='left'
    )

    # 清洗：负值→NaN，再按股票前向填充
    for c in available_cols:
        if c in processed.columns:
            processed[c] = pd.to_numeric(processed[c], errors='coerce')
            if c not in ('ROE_approx',):  # ROE 可正可负
                processed[c] = processed[c].mask(processed[c] <= 0)

    # 按股票前向填充缺失的基本面数据
    n_before = processed[available_cols].isna().sum().sum() if available_cols else 0
    for c in available_cols:
        if c in processed.columns:
            processed[c] = processed.groupby('股票代码')[c].ffill()
    n_after = processed[available_cols].isna().sum().sum() if available_cols else 0

    # 仍缺失的填0
    for c in available_cols:
        if c in processed.columns:
            processed[c] = processed[c].fillna(0.0)

    # 清理临时列
    if '_date_merge' in processed.columns:
        processed.drop(columns=['_date_merge'], inplace=True)

    print(f"  已合并基本面因子: {available_cols} (前向填充: {n_before}→{n_after} NaN)")
    return processed, available_cols


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# ============================================================
#  特征展平：60 天序列 → 单行向量（XGBRanker 输入）
# ============================================================

def flatten_sequences_to_xgb(data, features, sequence_length, flatten_days=None):
    """
    将时序 DataFrame 转换为 XGBRanker 需要的扁平特征矩阵。

    - 历史窗口 = sequence_length (60天，确保上下文)
    - 展平窗口 = flatten_days (默认10天，控制特征维度)
    - 市场状态特征已从输入中移除，改为在后处理阶段使用
    """
    import tempfile

    if flatten_days is None:
        flatten_days = config.get('xgb_flatten_days', 10)
    flatten_days = min(flatten_days, sequence_length)

    data = data.copy()
    data['日期'] = pd.to_datetime(data['日期'])
    data = data.sort_values(['instrument', '日期']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    date_list = sorted(data['日期'].unique())
    valid_dates = date_list[sequence_length - 1:]
    date2qid = {d: i for i, d in enumerate(valid_dates)}

    n_feat = len(features)
    feat_dim = flatten_days * n_feat  # 仅展平因子特征（市场状态特征已从输入中移除，改为后处理使用）

    # ── 第一遍：统计总样本数 ──
    print("正在统计样本数...")
    total_samples = 0
    for _, group in data.groupby('instrument', sort=False):
        n = len(group)
        group_dates = group['日期'].values
        for i in range(sequence_length - 1, n):
            if group_dates[i] in date2qid:
                total_samples += 1
    print(f"总样本数: {total_samples:,}")

    # ── 预分配 memmap（写到项目 output 目录而非系统临时目录） ──
    mmap_dir = config.get('output_dir', './model')
    os.makedirs(mmap_dir, exist_ok=True)
    tmpfile = tempfile.NamedTemporaryFile(suffix='.dat', delete=False, dir=mmap_dir)
    X = np.memmap(tmpfile.name, dtype=np.float32, mode='w+', shape=(total_samples, feat_dim))
    y = np.empty(total_samples, dtype=np.float32)
    qid = np.empty(total_samples, dtype=np.int32)

    # ── 预建每只股票的索引（日期 → 行号），用于快速查找 ──
    print("正在构建索引...")
    stock_groups = {}
    for stock, group in data.groupby('instrument', sort=False):
        group = group.set_index('日期').sort_index()
        stock_groups[stock] = {
            'feat': group[features].values.astype(np.float32),
            'label': group['label'].values.astype(np.float32),
            'dates': group.index.values,
        }

    # ── 第二遍：按日期顺序遍历，天然保证 qid 有序 ──
    print(f"正在展平时序特征（最后{flatten_days}天 × {n_feat}维 = {feat_dim:,}维，memmap 模式）...")
    write_pos = 0
    for q, d in enumerate(tqdm(valid_dates, desc="展平特征")):
        for stock, grp in stock_groups.items():
            idx = np.where(grp['dates'] == d)[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            if i < sequence_length - 1:
                continue
            # 只取最后 flatten_days 天展平
            seq = grp['feat'][max(0, i - flatten_days + 1): i + 1]
            # 补齐：若不足 flatten_days 天，前面补零
            if len(seq) < flatten_days:
                pad = np.zeros((flatten_days - len(seq), n_feat), dtype=np.float32)
                seq = np.vstack([pad, seq])
            flat = seq.flatten()
            X[write_pos] = flat
            y[write_pos] = grp['label'][i]
            qid[write_pos] = q
            write_pos += 1

    X_final = X[:write_pos]        # memmap 切片，不占额外 RAM
    y = y[:write_pos]
    qid = qid[:write_pos]

    print(f"展平完成：{write_pos:,} 个样本，{feat_dim:,} 维特征，{len(valid_dates)} 个交易组")
    return X_final, y, qid, None, None, valid_dates


# ============================================================
#  验证集划分
# ============================================================

def split_train_val_by_last_month(df, sequence_length, val_months=12):
    """按末尾 N 个月做验证集划分。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_start].copy()

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ============================================================
#  标签转换：连续超额收益 → 整数排名（XGBRanker 要求）
# ============================================================

def _continuous_labels_to_ranks(y, qid):
    """
    XGBRanker rank:pairwise 要求标签为整数。
    将每组 (qid) 内的连续标签按降序排名，最高收益→最高整数排名。

    例：y=[0.05, -0.02, 0.03], qid=[0,0,0] → rank=[2, 0, 1]
    同时返回原始连续标签，供后续 evaluate_xgb_model 计算实际收益指标。
    """
    y_rank = np.zeros_like(y, dtype=np.int32)
    for q in np.unique(qid):
        mask = qid == q
        group_y = y[mask]
        # argsort 两次得到 dense rank（0-based）
        order = np.argsort(group_y)                     # 升序：最低收益排最前
        rank = np.empty_like(order)
        rank[order] = np.arange(len(group_y))           # 0 = 最低，n-1 = 最高
        y_rank[mask] = rank
    return y_rank


# ============================================================
#  评估：复用 evaluation.py（不修改指标计算）
# ============================================================

def evaluate_xgb_model(model, X_val, y_val, qid_val, valid_dates, val_df, features,
                       scaler, sequence_length, k=5, min_gap=0.005):
    """
    在验证集上使用自定义指标评估 XGBRanker。
    步骤：按日期分组 → 对每组内的股票打分 → 计算 extended_metrics
    """
    import torch
    preds = model.predict(X_val)

    # 按 qid 分组，构建每日的 pred/true/mask
    daily_metrics = {
        'pred_return_sum': [], 'max_return_sum': [], 'random_return_sum': [],
        'ratio_pred': [], 'ratio_random': [], 'final_score': [],
        'topk_hit': [], 'spearman': [], 'win': [],
    }

    unique_qids = sorted(set(qid_val))
    num_total = 0
    num_valid = 0

    for q in unique_qids:
        mask = qid_val == q
        day_preds = torch.tensor(preds[mask], dtype=torch.float32)
        day_labels = torch.tensor(y_val[mask], dtype=torch.float32)
        n = len(day_preds)
        if n < k:
            continue
        num_total += 1

        # 按预测排序取 top k
        _, topk_idx = torch.topk(day_preds, k)
        topk_returns = day_labels[topk_idx]
        pred_sum = topk_returns.sum().item()

        _, true_topk_idx = torch.topk(day_labels, k)
        max_sum = day_labels[true_topk_idx].sum().item()
        random_sum = k * day_labels.mean().item()

        gap = max_sum - random_sum
        if abs(gap) < min_gap:
            continue

        num_valid += 1
        daily_metrics['pred_return_sum'].append(pred_sum)
        daily_metrics['max_return_sum'].append(max_sum)
        daily_metrics['random_return_sum'].append(random_sum)

        fs = (pred_sum - random_sum) / (gap + 1e-12) if abs(gap) > 1e-6 else 0.0
        daily_metrics['final_score'].append(fs)

        # TopK 命中
        true_set = set(true_topk_idx.numpy())
        pred_set = set(topk_idx.numpy())
        daily_metrics['topk_hit'].append(len(true_set & pred_set))

        # Spearman
        from evaluation import _spearman_rho_pytorch
        daily_metrics['spearman'].append(_spearman_rho_pytorch(day_preds, day_labels))

        # Win rate
        daily_metrics['win'].append(1.0 if topk_returns.mean().item() > day_labels.mean().item() else 0.0)

    n = num_valid
    metrics = {
        'final_score': np.mean(daily_metrics['final_score']) if n > 0 else 0.0,
        'topk_hit_rate': (np.mean(daily_metrics['topk_hit']) / k) if n > 0 else 0.0,
        'topk_hit_count': np.mean(daily_metrics['topk_hit']) if n > 0 else 0.0,
        'spearman_rho': np.mean(daily_metrics['spearman']) if n > 0 else 0.0,
        'win_rate': np.mean(daily_metrics['win']) if n > 0 else 0.0,
        'final_score_std': np.std(daily_metrics['final_score'], ddof=1) if n > 1 else 0.0,
        'pred_return_sum': np.mean(daily_metrics['pred_return_sum']) if n > 0 else 0.0,
        'valid_days_ratio': n / max(num_total, 1),
        'valid_days': n,
        'total_days': num_total,
    }
    return metrics


# ============================================================
#  单窗口训练函数（XGBRanker）
# ============================================================

def train_one_window(train_df, val_df, val_start, stockid2idx, num_stocks, config, output_dir):
    """
    XGBRanker 单窗口训练 + 评估。

    Args:
        train_df: 训练集 DataFrame
        val_df:   验证集 DataFrame
        val_start: 验证集起始日期
        stockid2idx: 股票代码映射
        num_stocks: 总股票数
        config: 配置字典
        output_dir: 输出目录

    Returns:
        best_score, extended_metrics
    """
    sequence_length = config['sequence_length']
    features_list = feature_columns_map[config['feature_num']]
    flatten_days = config.get('xgb_flatten_days', 10)

    # ── 特征工程 ──
    train_data, _ = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # ── 标准化 ──
    scaler = StandardScaler()
    for col_set in [train_data, val_data]:
        col_set[features_list] = col_set[features_list].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features_list)
    val_data = val_data.dropna(subset=features_list)
    train_data[features_list] = scaler.fit_transform(train_data[features_list])
    val_data[features_list] = scaler.transform(val_data[features_list])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # ── 合并基本面因子 ──
    # 优先使用时序基本面文件，回退到静态快照
    fundamental_path = os.path.join(config['data_path'], 'history_factors_nan.csv')
    if not os.path.exists(fundamental_path):
        fundamental_path = os.path.join(config['data_path'], 'hs300_fundamentals.csv')
    train_data, fund_cols = _merge_fundamentals(train_data, fundamental_path)
    if fund_cols:
        val_data, _ = _merge_fundamentals(val_data, fundamental_path)
        features_list = features_list + fund_cols

    # ── 因子IC筛选（仅当 selected_features 已配置时生效） ──
    if config.get('selected_features'):
        original_count = len(features_list)
        valid_features = [f for f in config['selected_features'] if f in features_list]
        features_list = valid_features
        print(f"特征筛选: {len(features_list)} 个显著因子 (从 {original_count} 缩减)")

    # ── 展平特征 ──
    X_train, y_train_cont, qid_train, _, _, valid_train_dates = flatten_sequences_to_xgb(
        train_data, features_list, sequence_length
    )
    X_val, y_val_cont, qid_val, val_sample_dates, val_sample_stocks, valid_val_dates = flatten_sequences_to_xgb(
        val_data, features_list, sequence_length
    )

    # ── 标签转换：连续超额收益 → 整数排名（XGBRanker rank:pairwise 要求） ──
    y_train = _continuous_labels_to_ranks(y_train_cont, qid_train)
    y_val = _continuous_labels_to_ranks(y_val_cont, qid_val)

    # ── XGBRanker 按组的样本数 ──
    train_groups = [np.sum(qid_train == q) for q in sorted(set(qid_train))]
    val_groups = [np.sum(qid_val == q) for q in sorted(set(qid_val))]

    print(f"\nXGBRanker 训练配置:")
    print(f"  训练样本: {len(X_train):,} 行，{X_train.shape[1]} 维特征")
    print(f"  训练组数: {len(train_groups)} 天")
    print(f"  验证样本: {len(X_val):,} 行")
    print(f"  验证组数: {len(val_groups)} 天")

    # ── 构建 XGBRanker ──
    xgb_params = {
        'max_depth': xgb_config['max_depth'],
        'learning_rate': xgb_config['learning_rate'],
        'n_estimators': xgb_config['n_estimators'],
        'subsample': xgb_config['subsample'],
        'colsample_bytree': xgb_config['colsample_bytree'],
        'reg_alpha': xgb_config['reg_alpha'],
        'reg_lambda': xgb_config['reg_lambda'],
        'min_child_weight': xgb_config['min_child_weight'],
        'objective': xgb_config['objective'],
        'eval_metric': xgb_config['eval_metric'],
        'ndcg_exp_gain': False,                         # 禁用指数增益（标签>31时必需）
        'verbosity': xgb_config['verbosity'],
        'n_jobs': xgb_config['n_jobs'],
        'tree_method': 'hist',
        'random_state': 42,
    }

    model = xgb.XGBRanker(**xgb_params)

    print("\n开始训练 XGBRanker ...")
    model.fit(
        X_train, y_train,
        qid=qid_train,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=20,
    )

    # ── 输出特征重要性 ──
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-20:][::-1]
    n_feat_per_day = len(features_list)
    print(f"\n特征重要性 Top20 (共{len(importance)}维, 每{n_feat_per_day}维=1天特征):")
    for rank, idx in enumerate(top_idx):
        day = idx // n_feat_per_day + 1
        f_idx = idx % n_feat_per_day
        label = f"T-{flatten_days - day + 1}天_{features_list[f_idx][:8]}"
        print(f"  {rank+1:2d}. {label}: {importance[idx]:.6f}")

    # ── 评估（使用原始连续收益标签，非整数排位） ──
    min_gap_val = config_extended.get('min_gap', 0.005)
    k_val = config_extended.get('eval_top_k', 5)
    extended_metrics = evaluate_xgb_model(
        model, X_val, y_val_cont, qid_val, valid_val_dates,
        val_data, features_list, scaler, sequence_length,
        k=k_val, min_gap=min_gap_val
    )

    best_score = extended_metrics.get('final_score', 0.0)

    # ── 保存模型 ──
    model_path = os.path.join(output_dir, 'best_model.json')
    model.save_model(model_path)
    # 同时保存 pkl（兼容 cross_val.py）
    joblib.dump(model, os.path.join(output_dir, 'best_model.pkl'))

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best final_score: {best_score:.6f}\n")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({**config, **xgb_config}, f, indent=4, ensure_ascii=False)

    print(f"\n模型已保存到: {model_path}")
    print(f"验证集最终得分 (final_score): {best_score:.6f}")

    eval_report = format_eval_report(extended_metrics)
    print(eval_report)
    with open(os.path.join(output_dir, 'eval_report.txt'), 'w', encoding='utf-8') as f:
        f.write(eval_report)

    return best_score, extended_metrics


# ============================================================
#  主程序
# ============================================================

def main():
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_path = config['data_path']
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str}, low_memory=False)

    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length'],
        val_months=config_extended.get('val_months', 12)
    )

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    print(f"全量数据范围: {full_df['日期'].min()} 到 {full_df['日期'].max()}")
    print(f"训练集范围: {train_df['日期'].min()} 到 {train_df['日期'].max()}")
    print(f"验证集范围: {val_df['日期'].min()} 到 {val_df['日期'].max()}")

    best_score, best_extended_metrics = train_one_window(
        train_df, val_df, val_start, stockid2idx, num_stocks, config, output_dir
    )

    print(f"\n{'#'*50}")
    print(f"  训练完成！最佳 final_score: {best_score:.6f}")
    print(f"{'#'*50}")
    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
