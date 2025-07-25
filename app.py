import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Import vnstock
from vnstock import Quote

st.title("Stock Price Prediction: Next-Day Close with ML (Live Data)")

# 1. User selects symbol and dates
symbol = st.text_input("Enter stock symbol (e.g. VCB):", value="VCB")
start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-07-25"))

if st.button("Fetch Data and Train Model"):
    # 2. Fetch data from vnstock API
    try:
        q = Quote(symbol=symbol)
        df = q.history(start=str(start_date), end=str(end_date), resolution='1D')
        st.success(f"Fetched {len(df)} rows of data for {symbol}")
        st.write(df.head())

        # 3. Prepare data
        df['close_next'] = df['close'].shift(-1)
        df = df.dropna()
        st.write(f"Number of rows after dropping NA: {len(df)}")

        # 4. Split features and target
        X = df[['open', 'high', 'low', 'close', 'volume']]
        y = df['close_next']

        # 5. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 6. Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 7. Predict
        predictions = model.predict(X_test)

        # 8. Evaluate
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R2 Score: {r2:.4f}")

        # 9. Visualize
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_test.values, label='Actual')
        ax.plot(predictions, label='Predicted')
        ax.legend()
        ax.set_title(f'Actual vs Predicted Next-Day Close Price ({symbol})')
        ax.set_xlabel('Test Sample')
        ax.set_ylabel('Close Price')
        st.pyplot(fig)

        # (Optional) Save model
        if st.button("Save Model"):
            joblib.dump(model, f"{symbol}_stock_model.pkl")
            st.success(f"Model saved as {symbol}_stock_model.pkl")

        # (Optional) Show feature importance
        importances = model.feature_importances_
        feature_names = X.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False)
        st.subheader("Feature Importance")
        st.write(feat_df)

    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")

else:
    st.info("Enter a stock symbol and select dates, then click the button above.")

st.caption("Made with ❤️ using vnstock, Streamlit & scikit-learn")
