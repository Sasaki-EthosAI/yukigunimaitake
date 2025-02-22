import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def sarima_2023_train_2024_test_single_model(df: pd.DataFrame, customer_code: str, product_code: int):
    # 1) 日付を datetime に変換
    df['shipment_date'] = pd.to_datetime(
        df['shipment_date'].astype(str),
        format='%Y%m%d',
        errors='coerce'
    )

    # 2) 日付ソート & customer_code, product_code でフィルタ
    df.sort_values('shipment_date', inplace=True)
    df_filtered = df.loc[
        (df['customer_code'].astype(str) == str(customer_code)) &
        (df['product_code'].astype(str) == str(product_code))
    ].copy()
    if df_filtered.empty:
        print("★ 該当データがありません。終了。")
        return np.nan

    # 3) インデックス化
    df_filtered.set_index('shipment_date', inplace=True)

    # 4) 学習(～2023-12-31) & テスト(2024-01-01～) に分割
    train = df_filtered.loc[df_filtered.index <= '2023-12-31', 'wholesale_price']
    test  = df_filtered.loc[df_filtered.index >= '2024-01-01', 'wholesale_price']
    if len(train) < 30 or len(test) == 0:
        print("★ 学習またはテストデータ不足")
        return np.nan

    # 5) SARIMAモデル定義 & 学習
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 365)
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # 6) テスト期間を一括予測 → get_forecast(steps=...)
    steps = len(test)  # テストデータのサンプル数
    forecast_res = results.get_forecast(steps=steps)
    predictions = forecast_res.predicted_mean

    # 予測結果のインデックスは整数になりがちなので、テスト期間の日付を割り当て
    predictions.index = test.index

    # 7) MSE計算
    mse = mean_squared_error(test, predictions)
    print(f"★ SARIMA MSE: {mse:.3f}")

    # 8) 可視化
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test, label='Test (Actual)', color='black')
    plt.plot(predictions.index, predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f"SARIMA(1,1,1)(1,1,1,12)\nTrain: ~2023/12/31, Test: 2024~\n"
              f"customer_code={customer_code}, product_code={product_code}")
    plt.xlabel("Date")
    plt.ylabel("wholesale_price")
    plt.legend()
    plt.grid()
    plt.show()

    return mse

# --- 使用例 ---
if __name__ == "__main__":
    df = pd.read_csv("分析用フォルダ/product_code=130049_customer_code=4721108.csv")
    mse_val = sarima_2023_train_2024_test_single_model(df, 4721108, 130049)
    print("最終MSE:", mse_val)
