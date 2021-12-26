# Lint as: python3
"""Tests for google3.location.lbs.quake.tools.geo_utils."""

import math
import unittest 


class GeoUtilsTest(unittest.TestCase)):

  def test_calculate_dist(self):
    lat1_degree, lng1_degree = 36.0, -122.0
    lat2_degree, lng2_degree = 37.0, -122.0

    dist_km = geo_utils.calculate_dist(lat1_degree,
                                       lng1_degree,
                                       lat2_degree,
                                       lng2_degree)

    self.assertAlmostEqual(dist_km, 111.19492664454764, places=4)

  def test_to_xyz(self):
    pt = (-123., 36.)
    x, y, z = geo_utils.to_xyz(pt)
    self.assertAlmostEqual(x, 0.44062223512712906, places=5)
    self.assertAlmostEqual(y, 0.678498742149937, places=5)
    self.assertAlmostEqual(z, -0.587785252292473, places=5)

  def test_arc_kdtree_count_neighbors(self):
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    tree = geo_utils.ArcKDTree(pts)
    n_pairs = tree.count_neighbors(tree, 0)
    self.assertEqual(n_pairs, 4)

  def test_arc_kdtree_query(self):
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    tree = geo_utils.ArcKDTree(pts)
    d, _ = tree.query((90, 0), k=4)
    circumference = 2 * math.pi * geo_utils.RADIUS_EARTH_KM
    self.assertAlmostEqual(d[0], circumference / 4.0, places=5)

  def test_arc_kdtree_query_ball_tree(self):
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    tree = geo_utils.ArcKDTree(pts)
    results = tree.query_ball_tree(tree, tree.circumference/4.)
    self.assertEqual(results, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

  def test_arc_kdtree_query_pairs(self):
    pts = [(0, 90), (0, 0), (180, 0), (0, -90)]
    tree = geo_utils.ArcKDTree(pts)
    results = tree.query_pairs(tree.circumference/4.)
    self.assertEqual(results, set([(0, 1), (1, 3), (2, 3), (0, 2)]))


if __name__ == '__main__':
  unittest.main()
