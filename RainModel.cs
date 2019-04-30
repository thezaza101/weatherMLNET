using System;
using System.IO;
using MLHelpers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace weatherMLNET
{
    public class WeatherData
    {
        [LoadColumn(0),ColumnName("RainfallMM")]
        public float RainfallMM;

        [LoadColumn(1),ColumnName("Temp3Pm")]
        public float Temp3Pm;

        [LoadColumn(2),ColumnName("Humidity3Pm")]
        public float Humidity3Pm;

        [LoadColumn(3),ColumnName("CloudOktas3Pm")]
        public float CloudOktas3Pm;

        [LoadColumn(4),ColumnName("WindDir3Pm")]
        public string WindDir3Pm;

        [LoadColumn(5),ColumnName("WindSpeed3Pm")]
        public float WindSpeed3Pm;

        [LoadColumn(6),ColumnName("MSLPressure3Pm")]
        public float MSLPressure3Pm;

        [LoadColumn(7),ColumnName("RainTomorrow")]
        public bool RainTomorrow;
    }

    public class RainPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;
        public float Probability { get; set; }


        public float Score { get; set; }
    }



    class RainModel : IMLTrainer
    {
        //Base path of the application
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        //Path to the data file
        private static string _dataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "data.csv");
        //Path to where the model will be saved
        public static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "rainModel.zip");
        
        //Reference to the MLContext
        private static MLContext _mlContext;
		//Reference to the pipeline of the model
        private static IEstimator<ITransformer> _pipeline;
        //Reference to the model    
        private static ITransformer _model;
        //Training data
        static IDataView _trainData;
        //Testing data
        static IDataView _testData;
		
		//Constructor for the iris model
        public RainModel()
        {
            _mlContext = new MLContext();
        }

        //Loads the data
        public void LoadData()
        {
            //Read all the data
            IDataView allData = _mlContext.Data.LoadFromTextFile<WeatherData>(path: _dataPath, hasHeader: false, separatorChar: ',');

            //split the data into test and training
            DataOperationsCatalog.TrainTestData splitData = _mlContext.Data.TrainTestSplit(allData, testFraction: 0.3,seed:1);
            _trainData = splitData.TrainSet;
            _testData = splitData.TestSet;  
        }

        //Data pre processing 
        public void BuildPipeline()
        {
            //by default the 'Label' column is considered to be the prediction target
            //Map the 'Species' column to the 'Label' column
            _pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName:"RainTomorrow")
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "WindDir3Pm", outputColumnName: "WindDir3PmFeature"))
          
            //Set the features to be used 
            .Append(_mlContext.Transforms.Concatenate("Features", "RainfallMM", "Humidity3Pm", "CloudOktas3Pm", "WindSpeed3Pm","WindDir3PmFeature","MSLPressure3Pm"))
            //cache the pipeline, this will make downstream processes faster
            .AppendCacheCheckpoint(_mlContext);
        }

        //Build and train the model
        public void BuildAndTrainModel()
        {
            //we can use the Stochastic Dual Coordinate Ascent (SDCA) maximum entropy classification model for our predictions
            //_pipeline = _pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            //_pipeline = _pipeline.Append(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            //_pipeline = _pipeline.Append(_mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"))
            
            _pipeline = _pipeline.Append(_mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));            
            //_pipeline = _pipeline.Append(_mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"))            
            //_pipeline = _pipeline.Append(_mlContext.BinaryClassification.Trainers.FastTree());            
            
            //Map the output to the PredictedLabel 
            //.Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //Train the model
            _model = _pipeline.Fit(_trainData);

            /*var txData = _model.Transform(_trainData);
            var x = txData.Schema;

            var featureColumns = txData.GetColumn<Single>(x[0]);
            foreach (var y in featureColumns)
            {
                System.Console.WriteLine(y);
            } */
        }

        //Evaluate the performance of the model
        public void EvaluateModel() 
        {
            var testMetrics = _mlContext.BinaryClassification.Evaluate(_model.Transform(_testData));
            System.Console.WriteLine($"Accuracy: {testMetrics.Accuracy:0.###}");
            System.Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:0.###}");
            //var testMetrics = _mlContext.MulticlassClassification.Evaluate(_model.Transform(_testData));
            //System.Console.WriteLine($"Micro Accuracy: {testMetrics.MicroAccuracy:0.###}");
            //System.Console.WriteLine($"Macro Accuracy: {testMetrics.MacroAccuracy:0.###}");
        }

        //Save the model to file
        public void SaveModelToFile(string pathToFile)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                _mlContext.Model.Save(_model,_trainData.Schema, fs);
            }
        }
        public void Predict(WeatherData wd)
        {
            System.Console.WriteLine("Predicting using trained model");
            System.Console.WriteLine(_mlContext.Model.CreatePredictionEngine<WeatherData, RainPrediction>(_model).Predict(wd).PredictedLabel);
        }
    }
}