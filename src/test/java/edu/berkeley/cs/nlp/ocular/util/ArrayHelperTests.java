package edu.berkeley.cs.nlp.ocular.util;

import static org.junit.Assert.*;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ArrayHelperTests {

	@Test
	public void test_sum_int() {
		assertEquals(225, ArrayHelper.sum(new int[] { 50, 0, 150, 25 }));
		assertEquals(25, ArrayHelper.sum(new int[] { 25 }));
		assertEquals(0, ArrayHelper.sum(new int[] { 0 }));
		assertEquals(0, ArrayHelper.sum(new int[] { 0, 0 }));
		assertEquals(0, ArrayHelper.sum(new int[0]));
	}

	@Test
	public void test_avg_int() {
		assertEquals(54.8, ArrayHelper.avg(new int[] { 50, 0, 150, 74, 0 }), 1e-9);
		assertEquals(67.5, ArrayHelper.avg(new int[] { 50, 150, 70, 0 }), 1e-9);
		assertEquals(90, ArrayHelper.avg(new int[] { 50, 150, 70 }), 1e-9);
		assertEquals(25.0, ArrayHelper.avg(new int[] { 25 }), 1e-9);
		assertEquals(0, ArrayHelper.avg(new int[] { 0 }), 1e-9);
		assertEquals(0, ArrayHelper.avg(new int[] { 0, 0 }), 1e-9);
		assertEquals(0, ArrayHelper.avg(new int[0]), 1e-9);
	}

	@Test
	public void test_sum_double() {
		assertEquals(2.25, ArrayHelper.sum(new double[] { 0.5, 0.0, 1.5, 0.25 }), 1e-9);
		assertEquals(0.25, ArrayHelper.sum(new double[] { 0.25 }), 1e-9);
		assertEquals(0.0, ArrayHelper.sum(new double[] { 0.0 }), 1e-9);
		assertEquals(0.0, ArrayHelper.sum(new double[] { 0.0, 0.0 }), 1e-9);
		assertEquals(0.0, ArrayHelper.sum(new double[0]), 1e-9);
	}

	@Test
	public void test_avg_double() {
		assertEquals(0.54, ArrayHelper.avg(new double[] { 0.5, 0.0, 1.5, 0.7, 0.0 }), 1e-9);
		assertEquals(0.675, ArrayHelper.avg(new double[] { 0.5, 1.5, 0.7, 0.0 }), 1e-9);
		assertEquals(0.9, ArrayHelper.avg(new double[] { 0.5, 1.5, 0.7 }), 1e-9);
		assertEquals(0.25, ArrayHelper.avg(new double[] { 0.25 }), 1e-9);
		assertEquals(0.0, ArrayHelper.avg(new double[] { 0.0 }), 1e-9);
		assertEquals(0.0, ArrayHelper.avg(new double[] { 0.0, 0.0 }), 1e-9);
		assertEquals(0.0, ArrayHelper.avg(new double[0]), 1e-9);
	}

	@Test
	public void test_min_int() {
		assertEquals(10, ArrayHelper.min(new int[] { 50, 10, 25, 150, 10, 25 }));
		assertEquals(25, ArrayHelper.min(new int[] { 25 }));
		assertEquals(20, ArrayHelper.min(new int[] { 20 }));
		assertEquals(20, ArrayHelper.min(new int[] { 20, 20 }));
		try {
			ArrayHelper.min(new int[0]);
			fail("exception expected");
		}
		catch(RuntimeException e) {
			// good
		}
	}

	@Test
	public void test_prepend() {
		{
			int[] b = ArrayHelper.prepend(0, new int[] { 1, 2, 3 });
			assertEquals(4, b.length);
			assertEquals(0, b[0]);
			assertEquals(1, b[1]);
			assertEquals(2, b[2]);
			assertEquals(3, b[3]);
		}
		{
			int[] b = ArrayHelper.prepend(0, new int[] {});
			assertEquals(1, b.length);
			assertEquals(0, b[0]);
		}
	}

	@Test
	public void test_append() {
		{
			Integer[] b = ArrayHelper.append(new Integer[] { 0, 1, 2 }, 3);
			assertEquals(4, b.length);
			assertEquals((int) 0, (int) b[0]);
			assertEquals((int) 1, (int) b[1]);
			assertEquals((int) 2, (int) b[2]);
			assertEquals((int) 3, (int) b[3]);
		}
		{
			Integer[] b = ArrayHelper.append(new Integer[] {}, 0);
			assertEquals(1, b.length);
			assertEquals((int) 0, (int) b[0]);
		}
	}

	@Test
	public void test_take() {
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 2);
			assertEquals(2, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 3);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 8);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.take(new int[] {}, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.take(new int[] {}, 2);
			assertEquals(0, b.length);
		}
	}

	@Test
	public void test_takeRight() {
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 2);
			assertEquals(2, b.length);
			assertEquals(2, b[0]);
			assertEquals(3, b[1]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 3);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 8);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] {}, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] {}, 2);
			assertEquals(0, b.length);
		}
	}
}
