using System;
using MLHelpers;

namespace weatherMLNET
{
    class Program
    {
        static void Main(string[] args)
        {
            RainModel model = new RainModel();
            model.LoadData();
            model.BuildPipeline();
            model.BuildAndTrainModel();
            model.EvaluateModel();
            model.SaveModelToFile(RainModel._modelPath);

            RainPredictor wageModel = new RainPredictor(RainModel._modelPath);

            WeatherData wd = new WeatherData(){
                RainfallMM = 0f,
                Temp3Pm = 13.20f,
                Humidity3Pm = 44.00f,
                CloudOktas3Pm = 2.00f,
                WindDir3Pm = "E",
                WindSpeed3Pm = 7.00f,
                MSLPressure3Pm = 1021.70f
            };
            
            model.Predict(wd);
            wageModel.Predict(wd);


            WeatherData wdx = new WeatherData(){
                RainfallMM = 0f,
                Temp3Pm = 30.20f,
                Humidity3Pm = 44.00f,
                CloudOktas3Pm = 8.00f,
                WindDir3Pm = "E",
                WindSpeed3Pm = 7.00f,
                MSLPressure3Pm = 1021.70f
            };
            
            model.Predict(wdx);
            wageModel.Predict(wdx);
        }
    }
}
