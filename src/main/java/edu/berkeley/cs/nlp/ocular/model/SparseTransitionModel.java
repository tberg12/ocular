package edu.berkeley.cs.nlp.ocular.model;

import java.util.Collection;

import tuple.Pair;

public interface SparseTransitionModel {
	public static interface TransitionState {
		public int getCharIndex();
		public int getOffset();
		public int getExposure();
		public Collection<Pair<TransitionState,Double>> forwardTransitions();
		public Collection<Pair<TransitionState,Double>> nextLineStartStates();
		public double endLogProb();
		public TransitionStateType getType();
		public String getLanguage();
	}
	public Collection<Pair<TransitionState,Double>> startStates(int d);
}
