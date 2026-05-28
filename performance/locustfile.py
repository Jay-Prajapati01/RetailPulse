from locust import HttpUser, between, task


class RetailPulseUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def open_home(self):
        self.client.get("/")

    @task(2)
    def open_forecasting(self):
        self.client.get("/?page=Forecasting")

    @task(2)
    def open_segmentation(self):
        self.client.get("/?page=Customer%20Segmentation")

    @task(1)
    def open_churn(self):
        self.client.get("/?page=Churn%20Analytics")

    @task(1)
    def open_inventory(self):
        self.client.get("/?page=Inventory%20Optimization")
