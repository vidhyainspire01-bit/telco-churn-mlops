"""Request/response contracts. Literal types reject unknown categories with a 422
instead of silently scoring garbage (OneHotEncoder would ignore it and the caller never knows)."""
from typing import List, Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
NetAddon = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NetAddon
    OnlineBackup: NetAddon
    DeviceProtection: NetAddon
    TechSupport: NetAddon
    StreamingTV: NetAddon
    StreamingMovies: NetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    predictions: List[Prediction]
