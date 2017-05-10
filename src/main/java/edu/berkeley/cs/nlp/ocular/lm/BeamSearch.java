package edu.berkeley.cs.nlp.ocular.lm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.cs.nlp.ocular.lm.SearchStateModel.SearchState;
import javafx.util.Pair;

public class BeamSearch {
	
	private class BackPointer {
		private SearchState state;
		private int bucketIndex;
		private int timeStep;
		
		public BackPointer (SearchState state, int bucketIndex, int t) {
			this.state = state;
			this.bucketIndex = bucketIndex;
			this.timeStep = t;
		}
		public SearchState getState() {
			return state;
		}
		public int getBucketIndex() {
			return bucketIndex;
		}
	}
	
	private class Bucket {		
		
		private HashMap<SearchState, Pair<BackPointer, Double>> states;
		
		public Bucket () {
			this.states = new HashMap<SearchState, Pair<BackPointer, Double>>();
		}
		
		public void addState (SearchState state, double transitionScore, double score, BackPointer bPointer) {
			Pair<BackPointer, Double> value = new Pair<BackPointer, Double>(bPointer, score+transitionScore); 
			Pair<BackPointer, Double> prevState = states.get(state);
			
			if (prevState != null) {
				if ((score+transitionScore) > prevState.getValue()) {
					states.replace(state, value);
				}
			}
			else {
				if (states.size() == 2*beamSize+1) {
					pruneStates();
				}
				states.put(state, value);
			}
		}
		
		public void pruneStates () {
			states = findMedianAndRemove(states);
		}
		
		public HashMap<SearchState, Pair<BackPointer, Double>> getStates() {
			return states;
		}
	}
	
	private final int beamSize;
	private final int numEpsilonBuckets;
	private final int maxLength;
	private SearchStateModel model;
	
	private List<List<Bucket>> buckets;
	
	private Pair<BackPointer, Double> bestPath;
	
	public BeamSearch (int beamSize, int numEpsilons, SearchStateModel model, int maxLength) {
		this.beamSize = beamSize;
		this.model = model;
		this.numEpsilonBuckets = numEpsilons;
		this.maxLength = maxLength;
		
		this.buckets = new ArrayList<List<Bucket>>();
				
		for (int i=0; i<=numEpsilonBuckets; i++) {
			this.buckets.add(new ArrayList<BeamSearch.Bucket>());
		}
	}
	
	private HashMap<SearchState, Pair<BackPointer, Double>> findMedianAndRemove(HashMap<SearchState, Pair<BackPointer, Double>> bucketStates) {
		
		double[] scores = new double[2*beamSize+1];
		int i = 0;
		
		for (Entry<SearchState, Pair<BackPointer, Double>> state : bucketStates.entrySet()) {
			scores[i++] = state.getValue().getValue();
		}
		
		while (i<scores.length) {
			scores[i++] = Double.NEGATIVE_INFINITY;
		}
		
		double median = MedianOfMedians.findMedian(scores);
		
		HashMap<SearchState, Pair<BackPointer, Double>> result = new HashMap<SearchState, Pair<BackPointer, Double>>();
		
		for (Entry<SearchState, Pair<BackPointer, Double>> state : bucketStates.entrySet()) {
			if (state.getValue().getValue() > median) {
				result.put(state.getKey(), state.getValue());
			}
		}
		
		return result;
	}
	
	private void checkBestPath (Bucket curBucket, int j, int t) {
		HashMap<SearchState, Pair<BackPointer, Double>> states = curBucket.getStates();
		
		for (Entry<SearchState, Pair<BackPointer, Double>> s : states.entrySet()) {
			double score = s.getKey().getEndScore() + s.getValue().getValue();
			
			if (score > bestPath.getValue()) {
				bestPath = new Pair<BackPointer, Double>(new BackPointer(s.getKey(), j, t), score);
			}
		}
	}
	
	public List<SearchState> startBeam () {		
		Bucket startBucket = new Bucket();
		startBucket.addState(model.startState(), 0.0, 0.0, null);
		this.bestPath = new Pair<>(new BackPointer(model.startState(), 0, 0), model.startState().getEndScore());
		
		return beam(startBucket);
	}
	
	private void fillEpsilonBuckets (int t) {
		for (int j=1; j<=numEpsilonBuckets; j++) {
			Bucket newBucket = new Bucket();
			Bucket prevBucket = buckets.get(j-1).get(t);
			
			for (Entry<SearchState, Pair<BackPointer, Double>> state : prevBucket.getStates().entrySet()) {
				List<Pair<SearchState, Double>> successors = state.getKey().getEpsilons();
				double curScore = state.getValue().getValue();
				
				BackPointer bPointer = new BackPointer(state.getKey(), j-1, t);
				
				for (Pair<SearchState, Double> s : successors) {
					newBucket.addState(s.getKey(), s.getValue(), curScore, bPointer);
				}
			}
			if (newBucket.getStates().size() > beamSize) {
				newBucket.pruneStates();
			}
			buckets.get(j).add(newBucket);
			checkBestPath(newBucket, j, t);
		}
	}
	
	private List<SearchState> getBestPath () {
		List<SearchState> bpStates = new ArrayList<SearchState>();
		BackPointer bp = bestPath.getKey();
		
		while (bp != null) {
			bpStates.add(bp.getState());
			bp = buckets.get(bp.bucketIndex).get(bp.timeStep).getStates().get(bp.getState()).getKey();
		}
		
		return bpStates;
	}
	
	private List<SearchState> beam (Bucket startBucket) {
		buckets.get(0).add(startBucket);
		fillEpsilonBuckets(0);
		
		for (int t=1; t<maxLength; t++) {
			Bucket newBucket = new Bucket();
			
			for (int j=0; j<=numEpsilonBuckets; j++) {
				Bucket prevBucket = buckets.get(j).get(t-1);
				
				for (Entry<SearchState, Pair<BackPointer, Double>> state : prevBucket.getStates().entrySet()) {
					List<Pair<SearchState, Double>> successors = state.getKey().getSuccessors();
					double curScore = state.getValue().getValue();
					
					BackPointer bPointer = new BackPointer(state.getKey(), j, t-1);
					
					for (Pair<SearchState, Double> s : successors) {
						newBucket.addState(s.getKey(), s.getValue(), curScore, bPointer);
					}
				}
			}
			if (newBucket.getStates().size() > beamSize) {
				newBucket.pruneStates();
			}
			buckets.get(0).add(newBucket);
			checkBestPath(newBucket, 0, t);
			fillEpsilonBuckets(t);
		}
		return getBestPath();
	}
}
