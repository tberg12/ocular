package edu.berkeley.cs.nlp.ocular.model.em;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;
import tberg.murphy.arrays.a;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class DenseBigramTransitionModel {
	private static double SPC_TO_SPC_SMOOTH = 1e-2;
	
	private double[] starts;
	private double[][] forwardTrans;
	private double[][] backwardTrans;
	
	public DenseBigramTransitionModel(LanguageModel lm) {
		int numC = lm.getCharacterIndexer().size();
		
		this.starts = new double[numC];
		for (int c=0; c<numC; ++c) {
			this.starts[c] = Math.log(lm.getCharNgramProb(new int[0], c));
		}
		
		this.forwardTrans = new double[numC][numC];
		for (int prevC=0; prevC<numC; ++prevC) {
			for (int c=0; c<numC; ++c) {
				this.forwardTrans[prevC][c] = Math.log(lm.getCharNgramProb(new int[] {prevC}, c));
			}
		}
		int spaceIndex = lm.getCharacterIndexer().getIndex(Charset.SPACE);
		a.scalei(this.forwardTrans[spaceIndex], (1.0 - SPC_TO_SPC_SMOOTH));
		this.forwardTrans[spaceIndex][spaceIndex] += SPC_TO_SPC_SMOOTH;
		
		this.backwardTrans = new double[numC][numC];
		for (int prevC=0; prevC<numC; ++prevC) {
			for (int c=0; c<numC; ++c) {
				this.backwardTrans[c][prevC] = this.forwardTrans[prevC][c];
			}
		}
	}
	
	public double endLogProb(@SuppressWarnings("unused") int c) {
		return 0.0;
	}
	
	public double startLogProb(int c) {
		return starts[c];
	}
	
	public double[] forwardTransitions(int c) {
		return forwardTrans[c];
		
	}
	
	public double[] backwardTransitions(int c) {
		return backwardTrans[c];
	}
}
