import os
import traceback

from rest_framework import viewsets, status, decorators
from rest_framework.response import Response
from .serializers import ModelSerializer, ModelDescriptionSerializer
from .models import Model, ModelDescription
from .review import get_review
from django.shortcuts import render
from multiprocessing import Process
import threading


def index(request):
    return render(request, "machine_learning/index.html")


class ModelViewSet(viewsets.ViewSet):

    def list(self, request):
        models = Model.objects.all()
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)

    def create(self, request):
        try:
            serializer = ModelSerializer(data=request.data)
            if serializer.is_valid():
                model = serializer.save()
                result = get_review(model.data_set.path)
                description = ModelDescription.objects.create(
                    model=model, description={})
                description_serializer = ModelDescriptionSerializer(description)
                return Response(
                    {"response": result, "model": serializer.data, "description": description_serializer.data})
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            traceback.print_exc()

    def destroy(self, request, pk):
        Model.objects.get(id=pk).delete()
        models = Model.objects.all()
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)


class ModelDescriptionViewSet(viewsets.ViewSet):

    def update(self, request, pk):
        description = ModelDescription.objects.get(id=pk)
        serializer = ModelDescriptionSerializer(description, request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FlaskModelViewSet(viewsets.ViewSet):
    # {"model": 12, "id_column": "Survived", "prediction_column": "PassengerId", "not_to_use_columns": ["Name"],
    #        "projectTitle": "test", "algo": "", "auto": 1, "unit": "", "description": 12}
    def list(self, request):
        models = Model.objects.all()
        for each_item in models:
            if each_item.model_type == "RG":
                each_item.model_type = "Regression"
            else:
                each_item.model_type = "Classification"
        serializer = ModelSerializer(models, many=True)
        return Response(serializer.data)

    def create(self, request):
        try:
            model_obj = Model.objects.get(id=request.data["model"])
            model_id = request.data["model"]
            model = model_obj.model_type
            description_obj = ModelDescription.objects.get(id=request.data["description"])
            train_csv_path = model_obj.data_set
            project_title = request.data["projectTitle"]
            auto = request.data["auto"]
            algo = request.data["algo"]
            model_obj.algorithm_name = algo
            model_obj.save()
            if algo == "":
                algo = "auto"
            if request.data["id_column"] == "":
                id_column = "null"
            else:
                id_column = request.data["id_column"]
            if request.data["prediction_column"] == "":
                predict = "null"
            else:
                predict = request.data["prediction_column"]
            if request.data["not_to_use_columns"]:
                drop = request.data["not_to_use_columns"]
            else:
                drop = ["null"]

            descriptions = description_obj.description
            unit = "null"
            label0 = "null"
            label1 = "null"
            if "unit" in request.data:
                if request.data["unit"] != "":
                    unit = request.data["unit"]
            if "label0" in request.data:
                if request.data["label0"] != "":
                    label0 = request.data["label0"]
            if "label1" in request.data:
                if request.data["label1"] != "":
                    label1 = request.data["label1"]
            # linux
            # p = Process(target=self.run,
            #                      args=(
            #                          train_csv_path, project_title, auto, id_column, predict, drop, descriptions, algo,
            #                          model_id,
            #                          model, unit, label0, label1))

            # windows
            p = threading.Thread(target=self.run,
                                 args=(
                                     train_csv_path, project_title, auto, id_column, predict, drop, descriptions, algo,
                                     model_id,
                                     model, unit, label0, label1))
            p.start()
            p.join()
            return Response(data={"message": "success"}, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()

    def run(self, train_csv_path, project_title, auto, id_column, predict, drop, descriptions, algo, model_id, model,
            unit, label0, label1):
        ""
        # linux
        os.system("kill -9 `lsof -t -i:8050`")
        # windows
#         os.system("npx kill-port 8050")
        if model in ['CL']:
            os.system(
                'python machine_learning/classifier_custom_explainer.py ' + str(
                    train_csv_path) + ' ' +'"'+ project_title +'"'+ ' ' + str(
                    auto) + ' ' +'"'+ id_column +'"'+ ' ' +'"'+ predict +'"'+ ' ' + '"' + str(drop) + '"' + ' ' +'"'+ str(
                    descriptions) +'"'+ ' ' + str(algo) + ' ' + str(model_id) + ' ' +'"'+ str(label0) +'"'+ ' ' +'"'+ str(label1) +'"')
        else:
            os.system(
                'python machine_learning/regression_custom_explainer.py ' + str(
                    train_csv_path) + ' ' +'"'+ project_title +'"'+ ' ' + str(
                    auto) + ' ' +'"'+ id_column +'"'+ ' ' +'"'+ predict +'"'+ ' ' + '"' + str(drop) + '"' + ' ' +'"'+ str(
                    descriptions) +'"'+ ' ' + str(algo) + ' ' + str(model_id) + ' ' +'"'+ str(unit) +'"')
