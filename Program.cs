using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			// create network
			int[] dimensions = { 3, 9, 1 };
			string filename = "patterns.csv";
			Network net = new Network ( dimensions, filename);

			// present training patterns and continue until error is reduced
			net.Train ();

			// re-test against training patters
			net.Test ();
		}
	}
}
