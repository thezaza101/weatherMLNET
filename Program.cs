using System;
using MLHelpers;

namespace weatherMLNET
{
    class Program
    {
        static void Main(string[] args)
        {
            IMLTrainer model = new RainModel();
            model.LoadData();
            model.BuildPipeline();
            model.BuildAndTrainModel();
            model.EvaluateModel();
        }
    }
}
