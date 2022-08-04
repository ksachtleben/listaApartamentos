from fastapi import APIRouter
from controllers.selling_controller import SellingController

def sellingRoutes(db):
    router = APIRouter()

    controller = SellingController(db)

    @router.get("/apartamentos")
    def sellingApartamentos():
        return controller.sellingApartamentos()

    @router.get("/neuralnetwork")
    def learningApartamentos():
        return controller.learningApartamentos()

    return router
