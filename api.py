from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Order(BaseModel):
    item_name: str
    quantity: int
    price: float
    is_spicy: bool = False
    toppings: List[str] = []

@app.post("/place-order")
async def place_order(order: Order):
    total_cost = order.quantity * order.price
    return {
        "message": f"Order for {order.item_name} placed successfully!",
        "total_bill": total_cost,
        "status": "Processing"
    }
