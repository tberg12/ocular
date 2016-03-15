package edu.berkeley.cs.nlp.ocular.util;

import java.util.Arrays;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ArrayHelper {

	public static int sum(int[] xs) {
		int result = 0;
		for (int x : xs) result += x;
		return result;
	}

	public static double sum(double[] xs) {
		double result = 0.0;
		for (double x : xs) result += x;
		return result;
	}

	public static double avg(int[] xs) {
		if (xs.length == 0) return 0.0;
		else return ((double)sum(xs)) / xs.length;
	}

	public static double avg(double[] xs) {
		if (xs.length == 0) return 0.0;
		else return sum(xs) / xs.length;
	}

	public static int max(int... xs) {
		if (xs.length == 0) throw new RuntimeException("ArrayHelper.max cannot be used on an empty array.");
		int v = Integer.MIN_VALUE;
		for (int x : xs) {
			if (x > v) v = x;
		}
		return v;
	}
	
	public static double max(double... xs) {
		if (xs.length == 0) throw new RuntimeException("ArrayHelper.max cannot be used on an empty array.");
		double v = Double.MIN_VALUE;
		for (double x : xs) {
			if (x > v) v = x;
		}
		return v;
	}
	
	public static int min(int... xs) {
		if (xs.length == 0) throw new RuntimeException("ArrayHelper.min cannot be used on an empty array.");
		int v = Integer.MAX_VALUE;
		for (int x : xs)
			if (x < v) v = x;
		return v;
	}

	public static double min(double... xs) {
		if (xs.length == 0) throw new RuntimeException("ArrayHelper.min cannot be used on an empty array.");
		double v = Double.MAX_VALUE;
		for (double x : xs)
			if (x < v) v = x;
		return v;
	}

	public static int[] prepend(int c, int[] vec1) {
		int[] result = new int[vec1.length + 1];
		if (vec1.length > 0) System.arraycopy(vec1, 0, result, 1, vec1.length);
		result[0] = c;
		return result;
	}

	public static <A> A[] append(A[] vec1, A c) {
		A[] result = Arrays.copyOf(vec1, vec1.length + 1);
		result[result.length - 1] = c;
		return result;
	}

	public static int[] take(int[] vec1, int n) {
		int n2 = Math.min(vec1.length, n);
		int[] result = new int[n2];
		if (vec1.length > 0) System.arraycopy(vec1, 0, result, 0, n2);
		return result;
	}

	public static int[] takeRight(int[] vec1, int n) {
		int n2 = Math.min(vec1.length, n);
		int[] result = new int[n2];
		if (vec1.length > 0) System.arraycopy(vec1, vec1.length - n2, result, 0, n2);
		return result;
	}

}
