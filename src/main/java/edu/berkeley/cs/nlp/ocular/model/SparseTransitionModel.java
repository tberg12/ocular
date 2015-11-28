package edu.berkeley.cs.nlp.ocular.model;

import java.util.Collection;

import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public interface SparseTransitionModel {
	public static interface TransitionState {
		public int getCharIndex();
		public int getOffset();
		public int getExposure();
		public Collection<Tuple2<TransitionState,Double>> forwardTransitions();
		public Collection<Tuple2<TransitionState,Double>> nextLineStartStates();
		public double endLogProb();
		public TransitionStateType getType();
		public String getLanguage();
	}
	public Collection<Tuple2<TransitionState,Double>> startStates();
}
