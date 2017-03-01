package edu.berkeley.cs.nlp.ocular.model.em;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;
import tberg.murphy.arrays.a;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class DenseBigramTransitionModel {
	private int numC;
	
	public DenseBigramTransitionModel(int numC) { 
		this.numC = numC;
	}
	
	public double endLogProb(@SuppressWarnings("unused") int c) {
		return 0.0;
	}
	
	public double startLogProb(int c) {
		return 0.0;
	}
	
	public double[] forwardTransitions(int c) {
		return a.zerosDouble(numC);
		
	}
	
	public double[] backwardTransitions(int c) {
		return a.zerosDouble(numC);
	}
}

