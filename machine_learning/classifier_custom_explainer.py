import ast
import json
import sqlite3
import sys

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingRegressor, \
    RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, BaggingRegressor, ExtraTreesRegressor

from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *
import dash_html_components as html
import dash_bootstrap_components as dbc
from explainerdashboard.custom import *
from explainerdashboard import ExplainerDashboard
import pandas as pd
import csv
from pathlib import Path

db_path = 'db.sqlite3'
mapping_json = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "LGBMClassifier": LGBMClassifier,
    "XGBClassifier": XGBClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "BaggingRegressor": BaggingRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor}


def prepare_model(drop, IDColumn):
    df = pd.read_csv(train_csv)
    has_header = csv.Sniffer().has_header(open(train_csv).read(2048))

    # id column set
    if (IDColumn != ""):
        IDColumn = IDColumn.replace(' ', '_')
        df.set_index(IDColumn, drop=True, inplace=True)
        df.index.name = IDColumn

    # predict to columns
    result = predict.replace(' ', '_')

    # convert list drop
    if drop != []:
        converter = lambda x: x.replace(' ', '_')
        drop = list(map(converter, drop))
        drop

    # space to underscore for all headers
    if has_header == False:
        df.columns = ['co_' + str(i + 1) for i in range(len(df.iloc[0].values))]
    df.columns = df.columns.str.replace(' ', '_')

    # drop unused columns
    if drop != []:
        df.drop(columns=drop, axis=1, inplace=True)

    cols = df.columns
    cols_dtypes = df.dtypes
    is_null = df.isnull().any()
    null_cols = []
    for col in cols:
        if is_null[col] == True:
            null_cols.append(col)
            if cols_dtypes[col] == 'float':
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    df.drop_duplicates(inplace=True)

    catCols = df.select_dtypes("object").columns
    nCols = df.select_dtypes(exclude='object').columns
    catCols = list(set(catCols))
    nCols = list(set(nCols))

    catCols = df.select_dtypes("object").columns
    catCols = list(set(catCols))
    df = pd.get_dummies(df, columns=catCols)

    df1 = df[[result]]
    df.drop([result], axis=1, inplace=True)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(df, df1, test_size=0.3, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test, catCols


def flask_main(x_train, x_test, y_train, y_test, catCols, model_id):
    if auto == 1:
        b = []
        a = [LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, XGBClassifier]

        for i in a:
            model = i().fit(x_train, y_train.values.ravel())

            # testing training accuracy
            from sklearn import metrics
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.metrics import average_precision_score
            from sklearn.metrics import roc_auc_score
            from sklearn.metrics import brier_score_loss

            y_pred = model.predict(x_test)
            x = metrics.accuracy_score(y_test, y_pred)
            y = balanced_accuracy_score(y_test, y_pred)
            z = average_precision_score(y_test, y_pred)
            z1 = roc_auc_score(y_test, y_pred)
            z2 = brier_score_loss(y_test, y_pred)
            b.append((x + y + z1) / 3 / 0.01)
            print(x, y, z, z1, z2)

        print(b)
        best_score = max(b)
        print("best_score ", str(best_score))
        index = b.index(best_score)

        def switch(index):
            if index == 0:
                return [LogisticRegression]
            elif index == 1:
                return [RandomForestClassifier]
            elif index == 2:
                return [GradientBoostingClassifier]
            elif index == 3:
                return [DecisionTreeClassifier]            
            elif index == 4:
                return [XGBClassifier]

        print(switch(index))
        automodel = switch(index)
        print("automodel", automodel)
        automodel_string = str(automodel[0]).split(".")[-1].replace("'>","")
        connpath = sqlite3.connect(db_path)
        with connpath as db:
            c = db.cursor()
            data = (str(best_score), model_id)
            sql_update_query = """Update machine_learning_model set overall_score = ? where id = ?"""
            c.execute(sql_update_query, data)
            db.commit()
            data = (str(automodel_string), model_id)
            sql_update_query = """Update machine_learning_model set algorithm_name = ? where id = ?"""
            c.execute(sql_update_query, data)
            db.commit()
        try:
            connpath.close()
        except:
            pass

        for i in automodel:
            model = i().fit(x_train, y_train.values.ravel())
            explainer = ClassifierExplainer(model, x_test, y_test,
                                            cats=catCols,
                                            descriptions=descriptions,
                                            labels=[label0, label1])

            explainer.plot_contributions(0)

            class CustomDashboard(ExplainerComponent):
                def __init__(self, explainer, name=None, **kwargs):
                    super().__init__(explainer, title="Impact")
                    self.shap_summary = ShapSummaryComponent(explainer, name=self.name + "summary",
                                                             hide_title=True, hide_depth=True)
                    self.precision = PrecisionComponent(explainer, name=self.name + "precision",
                                                        hide_cutoff=True, hide_binsize=True,
                                                        hide_binmethod=True, hide_multiclass=True,
                                                        hide_selector=True, hide_title=True, hide_footer=True,
                                                        cutoff=None, hide_depth=True)

                    self.featuredesc = FeatureDescriptionsComponent(explainer, name=self.name + "featuredesc",
                                                                    hide_title=True, hide_subtitle=True)
                    # self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
                    self.predictiontab = FeatureInputComponent(explainer, name=self.name + "predictiontab",
                                                               title="What If..",
                                                               subtitle="Adjust the column values to change the prediction",
                                                               hide_index=False, hide_depth=False, fill_row_first=True)
                    self.predictiongraph = ShapContributionsGraphComponent(explainer,
                                                                           name=self.name + "predictiongraph",
                                                                           hide_title=False,
                                                                           subtitle="How has each value contributed to the prediction?",
                                                                           hide_index=False, hide_depth=False,
                                                                           hide_sort=False,
                                                                           feature_input_component=self.predictiontab,
                                                                           hide_selector=False)
                    self.predictioncontrib = ShapContributionsTableComponent(explainer,
                                                                             name=self.name + "predictioncontrib",
                                                                             hide_title=False,
                                                                             subtitle="How has each value contributed to the prediction?",
                                                                             hide_index=False, hide_depth=False,
                                                                             hide_sort=False,
                                                                             feature_input_component=self.predictiontab,
                                                                             hide_selector=False)
                    self.predictionsum = ClassifierPredictionSummaryComponent(explainer,
                                                                              name=self.name + "predictionsum",
                                                                              **kwargs)
                    self.predictionsum1 = ImportancesComponent(explainer, name=self.name + "predictionsum1",
                                                               hide_type=True, hide_selector=True, hide_title=True,
                                                               hide_subtitle=True)

                def layout(self):
                    return dbc.Container([
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.H3("Feature Descriptions"),
                        #         self.featuredesc.layout()
                        #     ], style=dict(margin=20)),
                        #     ], style=dict(margin=20)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Column Impact"),
                                html.Div(
                                    "Analyze the impact each column has sorted from highest to lowest on the prediction."),
                                html.Div(f"{self.explainer.columns_ranked_by_shap()[0]} had the biggest impact"
                                         f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                                         f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predictionsum1.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Individual Value Impact"),
                                html.Div(
                                    "Explore the values from each column that have the greatest and least impact on the prediction."),
                                # self.predictiontab.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predictiontab.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         self.predictiongraph.layout()
                        #     ], width=4, style=dict(margin=20)),
                        #     ]),
                        dbc.Row([
                            dbc.Col(self.predictioncontrib.layout(), style=dict(margin=10)),
                            dbc.Col(self.predictiongraph.layout(), style=dict(margin=10)),
                            # dbc.Col([
                            #     self.predictiongraph.layout()
                            # ]),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.H3("Prediction Summary"),
                        #         self.predsum.layout()
                        #     ], style=dict(margin=20)),
                        #     ]),
                    ])

            class CustomPredictionsTab(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Impact Relationship")
                    self.shap_summary = ShapSummaryComponent(explainer, name=self.name + "summary",
                                                             hide_title=True, hide_subtitle=True)
                    self.shap_dependence = ShapDependenceComponent(explainer, name=self.name + "dependence",
                                                                   title="Impact Relationship",
                                                                   subtitle="Relationship between Feature value and Impact value")
                    self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Link between Columns & Impact"),
                                html.Div("Analyze the relationship each column has on the impact."),
                                html.Div(
                                    "Click on a column in the Impact graph to explore the visualization in the Impact Relationship graph below."),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.shap_summary.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.shap_dependence.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                    ])

            class Interactions(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Impact Interaction")
                    self.interaction = InteractionSummaryComponent(explainer, name=self.name + "interaction",
                                                                   hide_depth=True)
                    self.interactiond = InteractionDependenceComponent(explainer, name=self.name + "interactiond",
                                                                       hide_depth=True)
                    self.interactioncon = InteractionSummaryDependenceConnector(self.interaction, self.interactiond)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Deep Dive"),
                                html.Div("Explore the effect of"),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.interaction.layout()
                            ], style=dict(margin=10)),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Feature impact plot"),
                                self.interactiond.layout()
                            ], style=dict(margin=10)),
                        ]),
                    ])

            class Classif(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Classification Metrics")
                    self.predsum = ClassifierModelSummaryComponent(explainer, name=self.name + "predsum",
                                                                   hide_cutoff=True, cutoff=0.5, hide_selector=True)
                    self.interaction = ConfusionMatrixComponent(explainer, name=self.name + "interaction",
                                                                hide_cutoff=True, cutoff=0.5, hide_normalize=True,
                                                                hide_selector=True)
                    self.interactiond = PrecisionComponent(explainer, name=self.name + "interactiond", hide_cutoff=True,
                                                           cutoff=0.5, hide_binmethod=True, hide_binsize=True)
                    self.interaction1 = CumulativePrecisionComponent(explainer, name=self.name + "interaction1",
                                                                     hide_cutoff=True, cutoff=0.5, hide_percentile=True)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Evaluate Classification Model Metrics"),
                                html.Div("Hover onto each metric to understand the classification model performance"),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predsum.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Visualize Precision and Recall"),
                                html.Div(
                                    "Precision can be observed when True Positives (predicted " f"{self.explainer.labels[0]}" " and actual" f"{self.explainer.labels[0]}" ") are higher than False Positives " "(predicted " f"{self.explainer.labels[1]}" " and actual" f"{self.explainer.labels[0]}"")."),
                                html.Div(
                                    "Recall can be observed when True Positives are higher than False Negatives" "(predicted " f"{self.explainer.labels[0]}" " and actual" f"{self.explainer.labels[1]}"")."),
                                # self.interaction.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.interaction.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         self.interaction1.layout()
                        #     ], style=dict(margin=20)),
                        # ]),
                    ])
                    # ExplainerDashboard(explainer, [CustomDashboard, CustomPredictionsTab, Classif], boostrap=dbc.themes.FLATLY, title=projecttitle, plot_sample=1000, header_hide_selector=True).run()

            return explainer, CustomDashboard, CustomPredictionsTab, Classif
    else:
        b = []
        for i in algo:
            model = i().fit(x_train, y_train.values.ravel())

            from sklearn import metrics
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.metrics import average_precision_score
            from sklearn.metrics import roc_auc_score
            from sklearn.metrics import brier_score_loss

            y_pred = model.predict(x_test)
            x = metrics.accuracy_score(y_test, y_pred)
            y = balanced_accuracy_score(y_test, y_pred)
            z = average_precision_score(y_test, y_pred)
            z1 = roc_auc_score(y_test, y_pred)
            z2 = brier_score_loss(y_test, y_pred)
            b.append((x + y + z1) / 3 / 0.01)
            print(x, y, z, z1, z2)

            print(b)
            best_score = max(b)
            print("best_score", best_score)
            connpath = sqlite3.connect(db_path)
            with connpath as db:
                c = db.cursor()
                data = (str(best_score), model_id)
                sql_update_query = """Update machine_learning_model set overall_score = ? where id = ?"""
                c.execute(sql_update_query, data)
                db.commit()

            # model_obj = Model.objects.get(id=model_id)
            # model_obj.overall_score = best_score
            # model_obj.save()
            explainer = ClassifierExplainer(model, x_test, y_test,
                                            cats=catCols,
                                            descriptions=descriptions,
                                            labels=[label0, label1], shap_kwargs=dict(check_additivity=False))

            explainer.plot_contributions(0)

            class CustomDashboard(ExplainerComponent):
                def __init__(self, explainer, name=None, **kwargs):
                    super().__init__(explainer, title="Impact")
                    self.shap_summary = ShapSummaryComponent(explainer, name=self.name + "summary",
                                                             hide_title=True, hide_depth=True)
                    self.precision = PrecisionComponent(explainer, name=self.name + "precision",
                                                        hide_cutoff=True, hide_binsize=True,
                                                        hide_binmethod=True, hide_multiclass=True,
                                                        hide_selector=True, hide_title=True, hide_footer=True,
                                                        cutoff=None, hide_depth=True)

                    self.featuredesc = FeatureDescriptionsComponent(explainer, name=self.name + "featuredesc",
                                                                    hide_title=True, hide_subtitle=True)
                    # self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
                    self.predictiontab = FeatureInputComponent(explainer, name=self.name + "predictiontab",
                                                               title="What If..",
                                                               subtitle="Adjust the column values to change the prediction",
                                                               hide_index=False, hide_depth=False, fill_row_first=True)
                    self.predictiongraph = ShapContributionsGraphComponent(explainer,
                                                                           name=self.name + "predictiongraph",
                                                                           hide_title=False,
                                                                           subtitle="How has each value contributed to the prediction?",
                                                                           hide_index=False, hide_depth=False,
                                                                           hide_sort=False,
                                                                           feature_input_component=self.predictiontab,
                                                                           hide_selector=False)
                    self.predictioncontrib = ShapContributionsTableComponent(explainer,
                                                                             name=self.name + "predictioncontrib",
                                                                             hide_title=False,
                                                                             subtitle="How has each value contributed to the prediction?",
                                                                             hide_index=False, hide_depth=False,
                                                                             hide_sort=False,
                                                                             feature_input_component=self.predictiontab,
                                                                             hide_selector=False)
                    self.predictionsum = ClassifierPredictionSummaryComponent(explainer,
                                                                              name=self.name + "predictionsum",
                                                                              **kwargs)
                    self.predictionsum1 = ImportancesComponent(explainer, name=self.name + "predictionsum1",
                                                               hide_type=True, hide_selector=True, hide_title=True,
                                                               hide_subtitle=True)

                def layout(self):
                    return dbc.Container([
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.H3("Feature Descriptions"),
                        #         self.featuredesc.layout()
                        #     ], style=dict(margin=20)),
                        #     ], style=dict(margin=20)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Column Impact"),
                                html.Div(
                                    "Analyze the impact each column has sorted from highest to lowest on the prediction."),
                                html.Div(f"{self.explainer.columns_ranked_by_shap()[0]} had the biggest impact"
                                         f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                                         f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predictionsum1.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Individual Value Impact"),
                                html.Div(
                                    "Explore the values from each column that have the greatest and least impact on the prediction."),
                                # self.predictiontab.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predictiontab.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         self.predictiongraph.layout()
                        #     ], width=4, style=dict(margin=20)),
                        #     ]),
                        dbc.Row([
                            dbc.Col(self.predictioncontrib.layout(), style=dict(margin=10)),
                            dbc.Col(self.predictiongraph.layout(), style=dict(margin=10)),
                            # dbc.Col([
                            #     self.predictiongraph.layout()
                            # ]),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.H3("Prediction Summary"),
                        #         self.predsum.layout()
                        #     ], style=dict(margin=20)),
                        #     ]),
                    ])

            class CustomPredictionsTab(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Impact Relationship")
                    self.shap_summary = ShapSummaryComponent(explainer, name=self.name + "summary",
                                                             hide_title=True, hide_subtitle=True)
                    self.shap_dependence = ShapDependenceComponent(explainer, name=self.name + "dependence",
                                                                   title="Impact Relationship",
                                                                   subtitle="Relationship between Feature value and Impact value")
                    self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Link between Columns & Impact"),
                                html.Div("Analyze the relationship each column has on the impact."),
                                html.Div(
                                    "Click on a column in the Impact graph to explore the visualization in the Impact Relationship graph below."),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.shap_summary.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.shap_dependence.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                    ])

            class Interactions(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Impact Interaction")
                    self.interaction = InteractionSummaryComponent(explainer, name=self.name + "interaction",
                                                                   hide_depth=True)
                    self.interactiond = InteractionDependenceComponent(explainer, name=self.name + "interactiond",
                                                                       hide_depth=True)
                    self.interactioncon = InteractionSummaryDependenceConnector(self.interaction, self.interactiond)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Deep Dive"),
                                html.Div("Explore the effect of"),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.interaction.layout()
                            ], style=dict(margin=10)),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Feature impact plot"),
                                self.interactiond.layout()
                            ], style=dict(margin=10)),
                        ]),
                    ])

            class Classif(ExplainerComponent):
                def __init__(self, explainer, name=None):
                    super().__init__(explainer, title="Classification Metrics")
                    self.predsum = ClassifierModelSummaryComponent(explainer, name=self.name + "predsum",
                                                                   hide_cutoff=True, cutoff=0.5, hide_selector=True)
                    self.interaction = ConfusionMatrixComponent(explainer, name=self.name + "interaction",
                                                                hide_cutoff=True, cutoff=0.5, hide_normalize=True,
                                                                hide_selector=True)
                    self.interactiond = PrecisionComponent(explainer, name=self.name + "interactiond", hide_cutoff=True,
                                                           cutoff=0.5, hide_binmethod=True, hide_binsize=True)
                    self.interaction1 = CumulativePrecisionComponent(explainer, name=self.name + "interaction1",
                                                                     hide_cutoff=True, cutoff=0.5, hide_percentile=True)

                def layout(self):
                    return dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Evaluate Classification Model Metrics"),
                                html.Div("Hover onto each metric to understand the classification model performance"),
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.predsum.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Visualize Precision and Recall"),
                                html.Div(
                                    "Precision can be observed when True Positives (predicted " f"{self.explainer.labels[0]}" " and actual " f"{self.explainer.labels[0]}" ") are higher than False Positives " "(predicted " f"{self.explainer.labels[1]}" " and actual " f"{self.explainer.labels[0]}"")."),
                                html.Div(
                                    "Recall can be observed when True Positives are higher than False Negatives" "(predicted " f"{self.explainer.labels[0]}" " and actual " f"{self.explainer.labels[1]}"")."),
                                # self.interaction.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        dbc.Row([
                            dbc.Col([
                                self.interaction.layout()
                            ], style=dict(margin=10)),
                        ], style=dict(margin=10)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         self.interaction1.layout()
                        #     ], style=dict(margin=20)),
                        # ]),
                    ])

            # ExplainerDashboard(explainer, [CustomDashboard, CustomPredictionsTab, Classif], boostrap=dbc.themes.FLATLY, title=projecttitle, plot_sample=1000, header_hide_selector=True).run()

            return explainer, CustomDashboard, CustomPredictionsTab, Classif


if __name__ == '__main__':
    # variables to be passed
    print(sys.argv[0:])
    train_csv_path = sys.argv[1]
    projecttitle = sys.argv[2]
    catCols = ""
    if sys.argv[3] in [False, "False", "0", 0]:
        auto = 0
    elif sys.argv[3] in [True, "True", 1, "1"]:
        auto = 1
    else:
        auto = int(sys.argv[3])
    algo = sys.argv[8]
    if str(algo) in mapping_json:
        algo = [mapping_json[algo]]
    else:
        algo = [""]
    IDColumn = sys.argv[4]
    if IDColumn == "null":
        IDColumn = ""
    predict = sys.argv[5]
    if predict == "null":
        predict = ""
    model_id = sys.argv[9]
    drop = ast.literal_eval(sys.argv[6])
    if drop == ["null"]:
        drop = []
    descriptions = json.loads(sys.argv[7])
    label0 = sys.argv[10]
    if label0 == "null":
        label0 = ""
    label1 = sys.argv[11]
    if label1 == "null":
        label1 = ""
    train_csv = 'media/' + train_csv_path
    x_train, x_test, y_train, y_test, catCols = prepare_model(drop, IDColumn)
    explainer, CustomDashboard, CustomPredictionsTab, Classif = flask_main(x_train, x_test, y_train, y_test, catCols,
                                                                           model_id)
    ExplainerDashboard(explainer, [
        CustomDashboard, CustomPredictionsTab, Classif], boostrap=dbc.themes.FLATLY,
                       title=projecttitle, hide_poweredby=True, plot_sample=1000, header_hide_selector=True).run()
