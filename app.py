import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


st.set_page_config(
	page_title="Vendor Invoice Intelligence",
	page_icon=":bar_chart:",
	layout="centered",
)


st.title("Vendor Invoice Intelligence System")
st.write("Predict freight costs and identify invoices that may need manual review.")

mode = st.sidebar.radio(
	"Select a tool",
	("Freight Cost Prediction", "Invoice Risk Flagging"),
)

if mode == "Freight Cost Prediction":
	st.header("Freight Cost Prediction")
	quantity = st.number_input("Invoice Quantity", min_value=0.0, value=100.0)
	dollars = st.number_input("Invoice Dollars", min_value=0.0, value=5000.0)

	if st.button("Predict Freight Cost", type="primary"):
		result = predict_freight_cost({"Quantity": [quantity], "Dollars": [dollars]})
		if result is None:
			st.error("The freight model could not be loaded or used.")
		else:
			st.success(f"Predicted freight cost: ${result['Predicted_Freight'].iloc[0]:,.2f}")

else:
	st.header("Invoice Risk Flagging")
	invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0, value=50.0)
	invoice_dollars = st.number_input("Invoice Dollars", min_value=0.0, value=1200.0)
	freight = st.number_input("Freight Cost", min_value=0.0, value=45.0)
	total_item_quantity = st.number_input(
		"Total Item Quantity", min_value=0.0, value=50.0
	)
	total_item_dollars = st.number_input(
		"Total Item Dollars", min_value=0.0, value=1195.0
	)

	if st.button("Check Invoice Risk", type="primary"):
		input_data = {
			"invoice_quantity": [invoice_quantity],
			"invoice_dollars": [invoice_dollars],
			"Freight": [freight],
			"total_item_quantity": [total_item_quantity],
			"total_item_dollars": [total_item_dollars],
		}

		try:
			result = predict_invoice_flag(input_data)
			flag = int(result["Predicted_Flag"].iloc[0])
			if flag == 1:
				st.warning("Manual review recommended for this invoice.")
			else:
				st.success("Invoice appears safe.")
		except (RuntimeError, ValueError, OSError) as error:
			st.error(f"Invoice prediction failed: {error}")
