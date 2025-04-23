import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo

from sagemaker.processing import ProcessingInput, ProcessingOutput, PySparkProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.session import Session

# Create SageMaker session
pipeline_session = sagemaker.session.Session()

# Define role
role = sagemaker.get_execution_role()

#### Preprocessing Step ####
spark_processor = PySparkProcessor(
    base_job_name="preprocessing-step",
    framework_version="3.3",
    role=role,
    instance_type="ml.r5.4xlarge",
    instance_count=20,
    max_runtime_in_seconds=3600
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=spark_processor,
    inputs=[
        ProcessingInput(source="s3://your-bucket/raw-data/", destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="training-data", source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation-data", source="/opt/ml/processing/output/val")
    ],
    code="preprocess.py"
)

#### Training Step ####
xgb_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=pipeline_session.boto_region_name,
    version="1.5-1"
)

xgb_estimator = Estimator(
    image_uri=xgb_image_uri,
    instance_type="ml.m5.12xlarge",
    instance_count=2,
    role=role,
    output_path="s3://your-bucket/xgb-output/",
    base_job_name="xgb-risk-training",
    sagemaker_session=pipeline_session
)

xgb_estimator.set_hyperparameters(
    objective="reg:squarederror",
    num_round=500,
    max_depth=8,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

train_step = TrainingStep(
    name="TrainXGBoostModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["training-data"].S3Output.S3Uri,
            content_type="parquet"
        ),
        "validation": TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["validation-data"].S3Output.S3Uri,
            content_type="parquet"
        )
    }
)

#### Evaluation Step ####
eval_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["validation-data"].S3Output.S3Uri,
            destination="/opt/ml/processing/validation"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="evaluate.py",
    property_files=[evaluation_report]
)

#### Model Registration ####
model = Model(
    image_uri=train_step.properties.AlgorithmSpecification.TrainingImage,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role
)

register_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="RiskScoreXGBoostModelGroup",
        approval_status="PendingManualApproval",
        description="XGBoost regression model to predict risk score"
    )
)

#### Conditional Logic ####
condition_step = ConditionStep(
    name="CheckMAERange",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=eval_step.name, property_file=evaluation_report, json_path="mae"),
            right=35.0
        ),
        ConditionLessThanOrEqualTo(
            left=JsonGet(step_name=eval_step.name, property_file=evaluation_report, json_path="mae"),
            right=37.0
        )
    ],
    if_steps=[register_step],
    else_steps=[]
)

#### Define and Start Pipeline ####
pipeline = Pipeline(
    name="MyDataPipeline",
    steps=[preprocess_step, train_step, eval_step, condition_step],
    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=role)
pipeline.start()
