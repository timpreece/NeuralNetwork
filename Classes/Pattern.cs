﻿using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	/// <summary>
	/// Pattern of inputs and outputs e.g. for three neuron input and single output format would be 0.5, 0.5, 0.5, 1.0
	/// </summary>
	public class Pattern
	{
		private double[] _inputs;
		private double _output;

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Pattern"/> class.
		/// </summary>
		/// <param name="value">Value.</param>
		/// <param name="inputSize">Input size.</param>
		public Pattern (string value, int inputSize)
		{
			string[] line = value.Split (',');
			if (line.Length - 1 != inputSize) {
				throw new Exception ("Input does not match network configuration.");
			}

			_inputs = new double[inputSize];

			for (int i = 0; i < inputSize; i++) {
				_inputs [i] = double.Parse (line[i]);
			}

			_output = double.Parse(line[inputSize]); // last number is output

		}

		/// <summary>
		/// Gets the inputs.
		/// </summary>
		/// <value>The inputs.</value>
		public double[] Inputs
		{
			get { return _inputs; }
		}

		/// <summary>
		/// Gets the output.
		/// </summary>
		/// <value>The output.</value>
		public double Output
		{
			get { return _output; }
		}

	}
}

