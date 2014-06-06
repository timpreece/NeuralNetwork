﻿using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	public class Layer : List<Neuron>
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Layer"/> class for Input layer only
		/// </summary>
		/// <param name="size">Size.</param>
		public Layer (int size)
		{
			for (int i = 0; i < size; i++) {
				base.Add( new Neuron() );
			}
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Layer"/> class for Hidden and Output layer only
		/// </summary>
		/// <param name="size">Size.</param>
		/// <param name="layer">Layer.</param>
		/// <param name="rnd">Random.</param>
		public Layer(int size, Layer layer, Random rnd)
		{
			for (int i = 0; i < size; i++) {
				base.Add( new Neuron( layer, rnd ));
			}
		}

	}
}

