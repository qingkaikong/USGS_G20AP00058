# Lint as: python3
"""Tools related to geo-spatial operations."""

import math
import mgrs
import numpy
from scipy import inf
from scipy.spatial import cKDTree

def convert_latlng_to_mgrs(lats, lngs, MGRSPrecision = 1):
  '''
  Args:
    lats: list of latitudes.
    lngs: list of longitudes.
    MGRSPrecision: specifies the resolution:
      MGRSPrecision = 1, resolution is 10 km
      MGRSPrecision = 2, resolution is 1 km
      MGRSPrecision = 3, resolution is 100 m
      MGRSPrecision = 4, resolution is 10 m
      MGRSPrecision = 5, resolution is 1 m

  Returns:

  '''
  # Note, if you use MGRSPrecision = 1, it returns the center of the cell.
  # I tested using the map visualization: https://mappingsupport.com/p/coordinates-mgrs-google-maps.html
  m = mgrs.MGRS()
  inDegrees = True
  MGRSPrecision=MGRSPrecision
  n = len(lats)
  mgrs_token = list(map(m.toMGRS, lats, lngs, [True]*n, [MGRSPrecision]*n))
  
  return mgrs_token

def convert_mgrs_token_to_latlng(tokens):
  m = mgrs.MGRS()
  latlngs = map(m.toLatLon, tokens)
  return latlngs

def calculate_dist(lat1_degree: float, lng1_degree: float, lat2_degree: float,
                   lng2_degree: float) -> float:
  """Calculate distance between two points.

  This function calculate the distance on earth, assume earth is a sphere,
  and earth's radius is 6371 km. It is faster than the function in geopy.

  Args:
    lat1_degree: latitude of the first point.
    lng1_degree: longitude of the first point.
    lat2_degree: latitude of the second point.
    lng2_degree: longitude of the second point.

  Returns:
    dist_km: distance between these two points in km.
  """

  # Convert latitude and longitude to spherical coordinates in radians.
  degrees_to_radians = math.pi / 180.0

  # phi = 90 - latitude
  phi1 = (90.0 - lat1_degree) * degrees_to_radians
  phi2 = (90.0 - lat2_degree) * degrees_to_radians

  # theta = longitude
  theta1 = lng1_degree * degrees_to_radians
  theta2 = lng2_degree * degrees_to_radians

  # Compute spherical distance from spherical coordinates.

  # For two locations in spherical coordinates
  # (1, theta, phi) and (1, theta', phi')
  # cosine( arc length ) =
  #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
  # distance = rho * arc length
  cos_value = (
      math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
      math.cos(phi1) * math.cos(phi2))

  arc = math.acos(cos_value)
  dist_km = arc * 6371.
  return dist_km


def to_xyz(pt):
  """Converts a point's latitude and longitude to x,y,z.

  Args:
    pt: point, assumed to be in form (lng,lat)

  Returns:
    cartesian x, y, z in float
  """

  phi, theta = list(map(math.radians, pt))
  phi, theta = phi + math.pi, theta + (math.pi / 2)
  x = 1 * math.sin(theta) * math.cos(phi)
  y = 1 * math.sin(theta) * math.sin(phi)
  z = 1 * math.cos(theta)
  return x, y, z


# The following are code modified from PySAL kdtree
# https://github.com/pysal/libpysal/blob/master/libpysal/cg/kdtree.py

FLOAT_EPS = numpy.finfo(float).eps
RADIUS_EARTH_KM = 6371.0


class ArcKDTree(cKDTree):
  """Class to build a kdtree using geographical points.
  """

  def __init__(self, data, leafsize=10, radius=RADIUS_EARTH_KM):
    """KDTree using Arc Distance instead of Euclidean Distance.

    Returned distances are based on radius.
    Assumes data are Lng/Lat, does not account for geoids.
    For more information see docs for scipy.spatial.KDTree

    Args:
      data: array
        The data points to be indexed. This array is not copied, and so
        modifying this data will result in bogus results.Typically nx2.
      leafsize: int
        The number of points at which the algorithm switches over to
        brute-force. Has to be positive. Optional, default is 10.
      radius: float
        Radius of the sphere on which to compute distances.
        Assumes data in latitude and longitude.

    Examples
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> d,i = kd.query((90,0), k=4)
      >>> d
      array([10007.54339801, 10007.54339801, 10007.54339801, 10007.54339801])
      >>> circumference = 2*math.pi*RADIUS_EARTH_KM
      >>> round(d[0],5) == round(circumference/4.0,5)
      True
    """
    self.radius = radius
    self.circumference = 2 * math.pi * radius
    cKDTree.__init__(self, list(map(to_xyz, data)), leafsize)

  def _arcdist_to_linear(self, arc_dist, radius=RADIUS_EARTH_KM):
    """Converts arc distance to linear distance.

    Converts an arc distance (spherical) to a linear distance (R3) based on
    supplied radius.

    Args:
      arc_dist: arc distance on a sphere.
      radius: radius of the sphere, default is Earth radius in km.

    Returns:
      linear_dist: linear distance in R3.
    """

    c = 2 * math.pi * radius
    linear_dist = (2 - (2 * math.cos(math.radians(
        (arc_dist * 360.0) / c))))**(0.5)
    return linear_dist

  def _linear_to_arcdist(self, linear_dist, radius=RADIUS_EARTH_KM):
    """Convers a linear distance to arc distance.

    Converts a linear distance in the unit sphere (R3) to an arc distance
    based on supplied radius.

    Args:
      linear_dist: linear distance in km.
      radius: radius of the sphere, default is Earth radius in km

    Returns:
      arc_dist: arc distance on a sphere.
    """

    if linear_dist == float("inf"):
      return float("inf")
    elif linear_dist > 2.0:
      raise ValueError(
          "linear_dist, must not exceed the diameter of the unit sphere, 2.0")
    c = 2 * math.pi * radius
    a2 = linear_dist**2
    theta = math.degrees(math.acos((2 - a2) / (2.)))
    arc_dist = (theta * c) / 360.0
    return arc_dist

  def _toXYZ(self, x):
    if not issubclass(type(x), numpy.ndarray):
      x = numpy.array(x)
    if len(x.shape) == 2 and x.shape[1] == 3:  # assume point is already in XYZ
      return x
    if len(x.shape) == 1 and x.shape[0] == 3:  # assume point is already in XYZ
      return x
    elif len(x.shape) == 1:
      x = numpy.array(to_xyz(x))
    else:
      x = list(map(to_xyz, x))
    return x

  def count_neighbors(self, other, r, p=2):
    """Counts how many nearby pairs can be formed.

    See scipy.spatial.KDTree.count_neighbors.

    Args:
      other: KDTree instance, the other tree to draw points from.
      r: float or one-dimensional array of floats
         The radius to produce a count for. Multiple radii are searched with
         a single tree traversal.
      p: ignored, kept to maintain compatibility with scipy.spatial.KDTree.

    Returns:
      int or 1-D array of ints. The number of pairs. Note that this is
      internally stored in a numpy int, and so may overflow if very large (2e9).

    Examples
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> kd.count_neighbors(kd,0)
      4
      >>> circumference = 2.0*math.pi*RADIUS_EARTH_KM
      >>> kd.count_neighbors(kd,circumference/2.0)
      16
    """

    if r > 0.5 * self.circumference:
      raise ValueError(
          "r, must not exceed 1/2 circumference of the sphere (%f)." %
          self.circumference * 0.5)
    r = self._arcdist_to_linear(r, self.radius)
    return cKDTree.count_neighbors(self, other, r)

  def query(self, x, k=1, eps=0, p=2, distance_upper_bound=inf):
    """Queries the kd-tree for nearest neighbors.

    See scipy.spatial.KDTree.query for more details.

    Args:
      x: array-like, last dimension self.m
         query points are lng/lat.
      k: integer, The number of nearest neighbors to return.
      eps: nonnegative float, Return approximate nearest neighbors.
      p: ignored, kept to maintain compatibility with scipy.spatial.KDTree
      distance_upper_bound: nonnegative float, return only neighbors within
         this distance

    Returns:
      d: array or floats
         The distances to the nearest neighbors.
      i: array of integers
         The locations of the neighbors in self.data. i is the same shape as d.

    Examples
      >>> import numpy as np
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> d,i = kd.query((90,0), k=4)
      >>> d
      array([10007.54339801, 10007.54339801, 10007.54339801, 10007.54339801])
      >>> circumference = 2*math.pi*RADIUS_EARTH_KM
      >>> round(d[0],5) == round(circumference/4.0,5)
      True
      >>> d,i = kd.query(kd.data, k=3)
      >>> d2,i2 = kd.query(pts, k=3)
      >>> (d == d2).all()
      True
      >>> (i == i2).all()
      True
    """

    eps = self._arcdist_to_linear(eps, self.radius)
    if distance_upper_bound != inf:
      distance_upper_bound = self._arcdist_to_linear(distance_upper_bound,
                                                     self.radius)
    d, i = cKDTree.query(
        self,
        self._toXYZ(x),
        k,
        eps=eps,
        distance_upper_bound=distance_upper_bound)
    dims = len(d.shape)
    r = self.radius
    if dims == 0:
      return self._linear_to_arcdist(d, r), i
    if dims == 1:
      d = [self._linear_to_arcdist(x, r) for x in d]
    elif dims == 2:
      d = [[self._linear_to_arcdist(x, r) for x in row] for row in d]
    return numpy.array(d), i

  def query_ball_point(self, x, r, p=2, eps=0):
    """Finds all points within distance r of point(s) x.

       See scipy.spatial.KDTree.query_ball_point for more details.

    Args:
      x: array_like, shape tuple + (self.m,)
          The point or points to search for neighbors of.
      r: positive float. The radius of points to return.
      p: ignored, kept to maintain compatibility with scipy.spatial.KDTree
      eps: nonnegative float, optional. Approximate search.

    Returns:
      results: list or array of lists
        If x is a single point, returns a list of the indices of the neighbors
        of x. If x is an array of points, returns an object array of shape
        tuple containing lists of neighbors.

    Examples
      >>> import numpy as np
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> circumference = 2*math.pi*RADIUS_EARTH_KM
      >>> kd.query_ball_point(pts, circumference/4.)
      array([list([0, 1, 2]), list([0, 1, 3]), list([0, 2, 3]), list([1, 2,
      3])],
            dtype=object)
      >>> kd.query_ball_point(pts, circumference/2.)
      array([list([0, 1, 2, 3]), list([0, 1, 2, 3]), list([0, 1, 2, 3]),
             list([0, 1, 2, 3])], dtype=object)
    """

    eps = self._arcdist_to_linear(eps, self.radius)
    if r > 0.5 * self.circumference:
      raise ValueError(
          "r, must not exceed 1/2 circumference of the sphere (%f)." %
          self.circumference * 0.5)
    r = self._arcdist_to_linear(r, self.radius) + FLOAT_EPS * 3
    results = cKDTree.query_ball_point(self, self._toXYZ(x), r, eps=eps)
    return results

  def query_ball_tree(self, other, r, p=2, eps=0):
    """Finds all pairs of points whose distance is at most r.

    See scipy.spatial.KDTree.query_ball_tree.

    Args:
      other: KDTree instance, the tree containing points to search against.
      r: float, the maximum distance, has to be positive.
      p: ignored, kept to maintain compatibility with scipy.spatial.KDTree
      eps: loat, optional, approximate search.

    Returns:
      results: list of lists
        For each element self.data[i] of this tree, results[i] is a list of
        the indices of its neighbors in other.data.

    Examples
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> kd.query_ball_tree(kd, kd.circumference/4.) == [[0, 1, 2], [0, 1,
      3], [0, 2, 3], [1, 2, 3]]
      True
      >>> kd.query_ball_tree(kd, kd.circumference/2.) == [[0, 1, 2, 3], [0, 1,
      2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
      True
    """

    eps = self._arcdist_to_linear(eps, self.radius)
    if self.radius != other.radius:
      raise ValueError("Both trees must have the same radius.")
    if r > 0.5 * self.circumference:
      raise ValueError(
          "r, must not exceed 1/2 circumference of the sphere (%f)." %
          self.circumference * 0.5)
    r = self._arcdist_to_linear(r, self.radius) + FLOAT_EPS * 3
    results = cKDTree.query_ball_tree(self, other, r, eps=eps)
    return results

  def query_pairs(self, r, p=2, eps=0):
    """Finds all pairs of points within a distance.

    See scipy.spatial.KDTree.query_pairs for more details.

    Args:
      r: positive float, the maximum distance.
      p: ignored, kept to maintain compatibility with scipy.spatial.KDTree
      eps: loat, optional. Approximate search

    Returns:
      results: set
        Set of pairs (i,j), with i < j, for which the corresponding positions
        are close.

    Examples
      >>> pts = [(0,90), (0,0), (180,0), (0,-90)]
      >>> kd = Arc_KDTree(pts, radius = RADIUS_EARTH_KM)
      >>> kd.query_pairs(kd.circumference/4.) == set([(0, 1), (1, 3), (2, 3),
      (0, 2)])
      True
      >>> kd.query_pairs(kd.circumference/2.) == set([(0, 1), (1, 2), (1, 3),
      (2, 3), (0, 3), (0, 2)])
      True
    """

    if r > 0.5 * self.circumference:
      raise ValueError(
          "r, must not exceed 1/2 circumference of the sphere (%f)." %
          self.circumference * 0.5)
    r = self._arcdist_to_linear(r, self.radius) + FLOAT_EPS * 3
    results = cKDTree.query_pairs(self, r, eps=eps)
    return results
