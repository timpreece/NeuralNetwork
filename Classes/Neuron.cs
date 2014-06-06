using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	public class Neuron
	{
		private double _input;						// sum of inputs
		private double _bias;						// bias value
		private double _error; 						// sum of error
		private double _output = double.MinValue;	// output value
		private double _lrate = 0.5;				// learning rate
		private double _lambda = 5;					// steepness of sigmoid curve
		private List<Weight> _weights; 				// list of weights connected to inputs

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Neuron"/> class.
		/// </summary>
		public Neuron ()
		{
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Neuron"/> class.
		/// Randomises the weights between the Neurons in the Inputs layer.
		/// </summary>
		/// <param name="inputs">Inputs.</param>
		/// <param name="rnd">Random.</param>
		public Neuron (Layer inputs, Random rnd)
		{
			_weights = new List<Weight> ();
			foreach (Neuron input in inputs) {
				Weight w = new Weight ();
				w.Input = input;
				w.Value = (rnd.NextDouble () * 2) - 1;
				_weights.Add (w);
			}
		}

		/// <summary>
		/// Calculate the sum of inputs into Neuron
		/// </summary>
		public void Activate()
		{
			_input = 0;
			_error = 0;
			foreach (Weight w in _weights) {
				_input += w.Value * w.Input.Output;
			}
		}

		/// <summary>
		/// Collects the error.
		/// </summary>
		/// <param name="delta">Delta.</param>
		public void CollectError( double delta )
		{
			if (_weights != null) {
				_error += delta;
				foreach (Weight w in _weights) {
					w.Input.CollectError (_error * w.Value);
				}
			}
		}

		/// <summary>
		/// Errors the feedback.
		/// </summary>
		/// <returns>The feedback.</returns>
		/// <param name="input">Input.</param>
		/*public double ErrorFeedback (Neuron input)
		{
			Weight w = _weights.Find (delegate(Weight t) {
				return t.Input == input;
			});
			return _error * Derivative * w.Value;
		}*/

		/// <summary>
		/// Adjusts the weights.
		/// </summary>
		public void AdjustWeights()
		{
			for (int i = 0; i < _weights.Count; i++) {
				_weights [i].Value +=  _error * Derivative * _lrate * _weights [i].Input.Output;
			}
			_bias += _error * Derivative * _lrate;
		}

		/// <summary>
		/// Gets the derivative.
		/// </summary>
		/// <value>The derivative.</value>
		private double Derivative
		{
			get {
				double activation = Output;
				return activation * (1 - activation);
			}
		}

		/// <summary>
		/// Gets or sets the output.
		/// </summary>
		/// <value>The output.</value>
		public double Output
		{
			get {
				if (_output != double.MinValue) {
					return _output;
				}
				return 1 / (1 + Math.Exp (-_lambda * (_input + _bias)));
			}
			set {
				_output = value;
			}
		}
	}
}

