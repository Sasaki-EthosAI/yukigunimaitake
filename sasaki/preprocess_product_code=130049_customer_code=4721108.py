import pandas as pd

date_range = pd.date_range(start="2015-01-01",end="2025.2.1",freq="D")
df = pd.DataFrame(date_range,columns=["shipment_date"])

# データをデータフレームとして読み込む
csv = "../kawakura/Input/product_code=130049_customer_code=4721108.csv"
df_hand = pd.read_csv(csv)

# shipment_date を datetime 型に変換（フォーマットが "YYYYMMDD" なので適切に変換）
df_hand["shipment_date"] = pd.to_datetime(df_hand["shipment_date"].astype(str), format="%Y%m%d")

df_hand_grouped = df_hand.groupby("shipment_date", as_index=False).agg(
    shipment_quantity_bara=("shipment_quantity_bara", "sum"),
    wholesale_price=("wholesale_price", lambda x: (x * df_hand.loc[x.index, "shipment_quantity_bara"]).sum() / df_hand.loc[x.index, "shipment_quantity_bara"].sum()),
    product_code=("product_code", "first"),  # すべて同じなので最初の値を保持
    customer_code=("customer_code", "first")  # すべて同じなので最初の値を保持
)


# 2015-2025 の基準データフレームと結合（外部結合で全日付を保持）
df_merged = pd.merge(df, df_hand_grouped, on="shipment_date", how="left")

# 欠損値処理（shipment_quantity_bara: 0埋め, wholesale_price: 前日補完）
df_merged["shipment_quantity_bara"] = df_merged["shipment_quantity_bara"].fillna(0)
df_merged["wholesale_price"].fillna(method="ffill", inplace=True)

# 欠損値を前後のデータで補完し、それでも欠損がある場合は最も頻度の高い値を入れる
most_common_product_code = df_hand["product_code"].mode()[0]
most_common_customer_code = df_hand["customer_code"].mode()[0]

df_merged["product_code"] = df_merged["product_code"].fillna(method="ffill").astype(int).astype(str)
df_merged["customer_code"] = df_merged["customer_code"].fillna(method="ffill").astype(int).astype(str)

output_file = "product_code=130049_customer_code=4721108_daily_preprocessed.csv"
df_merged.to_csv(output_file,index=False,encoding="shift_jis")

print(f"統合完了: {output_file} に保存されました。")