using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	/// <summary>
	/// Network.
	/// Based on http://dynamicnotions.blogspot.co.uk/2008/09/training-neural-networks-using-back.html
	/// </summary>
	public class Network
	{
		private int _numInput; // dimensions of network
		private int _numHidden;
		private int _numOutput;

		private Layer _input; // layers of neurons in network
		private Layer _hidden;
		private Neuron _output;

		private List<Pattern> _patterns; // training patterns

		private Random _rnd; // random seed 

		private int _epoch;
		private int _restartAfter = 500; // max epochs before restart (stuck in local minima)

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		/// <param name="numInput">Number input Neurons.</param>
		/// <param name="numHidden">Number hidden Neurons.</param>
		/// <param name="numOutput">Number output Neurons.</param>
		public Network (int numInput, int numHidden, int numOutput, Random random)
		{
			_numInput = numInput;
			_numHidden = numHidden;
			_numOutput = numOutput;
			_rnd = random;
		}
			
		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		public void LoadTrainingPatterns()
		{
			_patterns = new List<Pattern> ();
			StreamReader file = File.OpenText ("patterns.csv");
			Console.WriteLine( "Loading training pattern" );
			while (!file.EndOfStream) {
				string line = file.ReadLine ();
				_patterns.Add( new Pattern( line, _numInput ));
				Console.WriteLine( line );
			}
			file.Close();
		}
		
		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		public void Initialise()
		{
			_input = new Layer (_numInput);
			_hidden = new Layer (_numHidden, _input, _rnd);
			_output = new Neuron( _hidden, _rnd);
			_epoch = 0;
			Console.WriteLine ("Network is initialised");
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		public void Train()
		{
			double error;
			do {
				error = 0;
				foreach (Pattern pattern in _patterns) {
					double delta = pattern.Output - Activate (pattern); // calculate delta error between activated output and target output
					AdjustWeights (delta); // back propogate error through network
					error += Math.Pow (delta, 2); // add error squared
				}
				Console.WriteLine (" Epoch{0}\tError {1:0.000}", _epoch, error);

				if( ++_epoch > _restartAfter ) 
					Initialise();

			} while(error > 0.1);

		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		public void Test()
		{
			Console.WriteLine ("Testing network");
			foreach (Pattern pattern in _patterns) {
				Console.WriteLine ("Input {0}, {1}\tOutput {2:0.000}", pattern.Inputs[0], pattern.Inputs[1], Activate (pattern));
			}
		}

		/// <summary>
		/// Activate the specified pattern.
		/// </summary>
		/// <param name="pattern">Pattern.</param>
		private double Activate( Pattern pattern )
		{
			// set the input layer to current pattern input values
			for (int i = 0; i < pattern.Inputs.Length; i++) {
				_input[i].Output = pattern.Inputs[i];
			}

			// propogate input through hidden layer
			foreach (Neuron neuron in _hidden) {
				neuron.Activate ();
			}

			// propogate hidden through to output neuron
			_output.Activate ();

			return _output.Output;
		}

		/// <summary>
		/// Adjusts the weights.
		/// </summary>
		/// <param name="delta">Delta.</param>
		private void AdjustWeights( double delta )
		{
			_output.AdjustWeights (delta);
			foreach (Neuron neuron in _hidden) {
				neuron.AdjustWeights (_output.ErrorFeedback (neuron));
			}
		}

	}
}

