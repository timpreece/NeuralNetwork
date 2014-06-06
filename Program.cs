using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			Network _network = new Network (2,2,1, new Random ());

			_network.LoadTrainingPatterns ();
			_network.Initialise ();
			_network.Train ();
			_network.Test ();
		}
	}
}
