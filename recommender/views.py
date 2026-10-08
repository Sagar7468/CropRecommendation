from django.shortcuts import render
from .ml_model import recommend_crops


def home(request):

    result = None

    if request.method == "POST":

        location = request.POST.get("location")

        latitude = float(request.POST.get("latitude"))
        longitude = float(request.POST.get("longitude"))

        N = float(request.POST.get("N"))
        P = float(request.POST.get("P"))
        K = float(request.POST.get("K"))
        ph = float(request.POST.get("ph"))

        month = request.POST.get("month")

        temperature_change = float(
            request.POST.get("temperature_change")
        )

        rainfall_change_percent = float(
            request.POST.get("rainfall_change_percent")
        )

        result = recommend_crops(
            latitude=latitude,
            longitude=longitude,
            N=N,
            P=P,
            K=K,
            ph=ph,
            month=month,
            temperature_change=temperature_change,
            rainfall_change_percent=rainfall_change_percent
        )

        result["location"] = location

    return render(
        request,
        "recommender/home.html",
        {"result": result}
    )
