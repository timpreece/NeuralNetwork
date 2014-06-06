using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
	/// <summary>
	/// Network.
	/// Based on http://dynamicnotions.blogspot.co.uk/2008/09/training-neural-networks-using-back.html
	/// </summary>
	public class Network : List<Layer>
	{
		private int[] _dimensions;
		private List<Pattern> _patterns; // training patterns

		private int _restartAfter = 500; // max epochs before restart (stuck in local minima)
		private int _epoch;

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		/// <param name="dimensions">Dimensions.</param>
		/// <param name="filename">Filename.</param>
		public Network (int[] dimensions, string filename)
		{
			_dimensions = dimensions;
			Initialise ();
			LoadTrainingPatterns (filename);
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		public void Initialise()
		{
			base.Clear ();
			// create input layer of neurons
			base.Add( new Layer( _dimensions[0] ) );
			// create all remaining layers of neurons
			for(int i = 1; i < _dimensions.Length; i++){
				base.Add( new Layer( _dimensions[i], base[i-1], new Random() ) );
			}
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="NeuralNetwork.Network"/> class.
		/// </summary>
		private void LoadTrainingPatterns(string filename)
		{
			_patterns = new List<Pattern> ();
			StreamReader file = File.OpenText (filename);
			while (!file.EndOfStream) {
				string line = file.ReadLine ();
				_patterns.Add( new Pattern( line, Inputs.Count, Outputs.Count ));
				Console.WriteLine( line );
			}
			file.Close();
		}

		/// <summary>
		/// Train this instance.
		/// </summary>
		public void Train()
		{
			double error;
			_epoch = 0;

			do {
				error = 0;
				foreach (Pattern pattern in _patterns) {

					Activate( pattern ); // present next training pattern

					for( int i = 0; i < Outputs.Count; i++){
						double delta = pattern.Outputs[i] - Outputs[i].Output;
						// capture this error into output neuron
						Outputs[i].CollectError( delta );
						// sum the squared error across all output neurons
						error += Math.Pow( delta, 2 );
					}
					AdjustWeights (); // back propogate error through network
				}

				Console.WriteLine (" Epoch{0}\tError {1:0.000}", _epoch, error);

				if( ++_epoch > _restartAfter ) 
					Initialise();

			} while(error > 0.05);
		}

		/// <summary>
		/// Test this instance.
		/// </summary>
		public void Test()
		{
			Console.WriteLine ("Testing network");
			foreach (Pattern pattern in _patterns) {
				Activate (pattern);

				String testString = "Input ";
				for (int i = 0; i < Inputs.Count; i++) {
					testString += String.Format("{0} ", Inputs [i].Output ); 
				}

				testString += " Output ";
				for (int i = 0; i < Outputs.Count; i++) {
					testString += String.Format("{0:F2} ", Outputs [i].Output ); 
				}

				Console.WriteLine (testString);
			}
		}

		/// <summary>
		/// Activate the specified pattern.
		/// </summary>
		/// <param name="pattern">Pattern.</param>
		private void Activate( Pattern pattern )
		{
			// set the input layer to current pattern input values
			for (int i = 0; i < Inputs.Count; i++) {
				Inputs[i].Output = pattern.Inputs[i];
			}
			// propogate input through all remaining layers
			for (int i = 1; i < base.Count; i++) {
				foreach (Neuron neuron in base[i]) {
					neuron.Activate ();
				}
			}
		}

		/// <summary>
		/// Adjusts the weights.
		/// </summary>
		private void AdjustWeights()
		{
			// go backwards through layers excluding input layer
			for (int i = base.Count - 1; i > 0; i--) {
				foreach (Neuron neuron in base[i]) {
					neuron.AdjustWeights ();
				}
			}
		}

		/// <summary>
		/// Gets the inputs.
		/// </summary>
		/// <value>The input Layer of neurons</value>
		private Layer Inputs
		{
			get { return base[0]; }
		}

		/// <summary>
		/// Gets the outputs.
		/// </summary>
		/// <value>The output Layer of neurons</value>
		private Layer Outputs
		{
			get { return base[ base.Count - 1]; }
		}
	}
}

