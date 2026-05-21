"""
app.py —— UrbanAir 完整版 Streamlit 应用 (骨架)
运行: streamlit run app.py

模块:
  1. AQI地图 —— 交互式Folium地图 + 监测站标注
  2. 模型对比 —— 柱状图/雷达图 7种模型指标
  3. 模态分析 —— SHAP + 消融实验可视化
  4. 预测器 —— 选点预测PM2.5 (迁移自MVP 06_app.py)
"""
import streamlit as st

st.set_page_config(
    page_title="UrbanAir",
    page_icon="🌍",
    layout="wide",
)

# ============================================================
# Sidebar 导航
# ============================================================
st.sidebar.title("🌍 UrbanAir")
st.sidebar.caption("城市环境质量多模态评估系统")

page = st.sidebar.radio(
    "导航",
    ["📊 模型对比", "🗺️ AQI地图", "🔬 模态分析", "🔮 预测器"],
)

# ============================================================
# 页面路由
# ============================================================
if page == "🗺️ AQI地图":
    st.title("🗺️ 城市AQI监测地图")
    st.info("TODO: 集成 Folium 交互地图 + 监测站标注 + AQI热力图")

elif page == "📊 模型对比":
    st.title("📊 模型性能对比")
    st.info("TODO: 集成 Plotly 柱状图/雷达图 + 7种模型指标对比")
    # 子区域: RMSE对比 | 训练时间对比 | 参数量对比 | 消融实验

elif page == "🔬 模态分析":
    st.title("🔬 模态贡献分析")
    st.info("TODO: 集成 SHAP 特征重要性 + 消融实验堆叠图 + 注意力权重热力图")

elif page == "🔮 预测器":
    st.title("🔮 空气质量预测器")
    st.info("TODO: 集成 MVP 06_app.py 的预测逻辑 + 卫星图展示 + Grad-CAM")

# ============================================================
# Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("TJU 数据分析与挖掘 · 期末项目 · 2024-2025")


if __name__ == "__main__":
    pass  # streamlit run 调用
