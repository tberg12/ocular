package edu.berkeley.cs.nlp.ocular.eval;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface EMIterationEvaluator {
	public void evaluate(int iter, Document doc, TransitionState[][] decodeStates, int[][] decodeWidths);

	/**
	 * No-op version of an evaluator 
	 */
	public static class NoOpEMIterationEvaluator implements EMIterationEvaluator {
		public void evaluate(int iter, Document doc, TransitionState[][] decodeStates, int[][] decodeWidths) {}
	}
}
