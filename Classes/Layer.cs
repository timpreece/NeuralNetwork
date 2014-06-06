using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	public class Layer : List<Neuron>
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Layer"/> class.
		/// </summary>
		/// <param name="size">Size.</param>
		public Layer (int size)
		{
			for (int i = 0; i < size; i++) {
				base.Add( new Neuron() );
			}
		}
		/// blah

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Layer"/> class.
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

