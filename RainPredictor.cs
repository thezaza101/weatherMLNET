using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace weatherMLNET
{
    //Data model of iris.txt
    public class RainPredictor
    {
        ITransformer _model;
        private static MLContext _mlContext;
        //Reference to the PredictionEngine for the iris model       
        private static PredictionEngine<WeatherData, RainPrediction> _predEngine;

        public RainPredictor(string pathToModel)
        {
            _mlContext = new MLContext();

            DataViewSchema intpuSchema_variable;
            using (var stream = File.OpenRead(pathToModel))
                _model = _mlContext.Model.Load(stream,out intpuSchema_variable);
            _predEngine = _mlContext.Model.CreatePredictionEngine<WeatherData, RainPrediction>(_model);
        }

        public void Predict(WeatherData wd)
        {
            System.Console.WriteLine("Predicting using loaded *trained* model");
            bool value = _predEngine.Predict(wd).PredictedLabel;
            System.Console.WriteLine(value);            
        }

    }
}