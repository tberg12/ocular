package edu.berkeley.cs.nlp.ocular.lm;

import java.util.List;

import javafx.util.Pair;

public interface BeamState {
	public List<Pair<BeamState,Double>> getSuccessors();
}
