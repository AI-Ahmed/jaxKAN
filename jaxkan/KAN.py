from typing import List, Union, Literal
from jax import numpy as jnp
import jax

from flax import nnx

from .layers import get_layer


class KAN(nnx.Module):
    """
    Kolmogorov-Arnold Network (KAN) implementation using JAX and Flax NNX.
    
    KANs are neural networks that replace traditional linear transformations followed by 
    fixed activation functions with learnable univariate functions on the edges of the network.
    Based on the Kolmogorov-Arnold representation theorem, KANs can represent any multivariate
    continuous function as a composition of univariate functions and addition operations.
    
    This implementation provides multiple basis function types including B-splines, 
    Chebyshev polynomials, Legendre polynomials, Fourier series, radial basis functions,
    and sine-based activations. Each layer can adaptively refine its grid/basis functions
    during training for improved approximation capabilities.

    Parameters
    ----------
    layer_dims : List[int]
        Defines the network architecture in terms of nodes per layer. 
        E.g., [4,5,1] creates a network with 2 layers: first layer has 4 inputs 
        and 5 outputs, second layer has 5 inputs and 1 output.
    layer_type : str, default='base'
        Type of KAN layer to use throughout the network. Available options:
        
        - 'base': Original spline-based KAN layer with residual connections
        - 'spline': Efficient spline-based implementation 
        - 'chebyshev': Chebyshev polynomial basis functions
        - 'legendre': Legendre polynomial basis functions
        - 'fourier': Fourier series basis functions
        - 'rbf': Radial basis function layer
        - 'sine': Sine-based basis functions
    required_parameters : dict or None, default=None
        Dictionary containing parameters required for the chosen layer type.
        Required parameters by layer type:
        
        **Base/Spline layers:**
        - k : int, spline order (typically 3)
        - G : int, number of grid intervals
        - grid_range : tuple, initial grid range, default=(-1, 1)
        - grid_e : float, grid uniformity parameter, default=0.05
        
        **Polynomial layers (Chebyshev/Legendre):**
        - D : int, polynomial degree
        - flavor : str, implementation variant ('default', 'exact')
        
        **Fourier layer:**
        - D : int, number of Fourier modes
        - smooth_init : bool, smoothening initialization, default=True
        
        **RBF layer:**
        - D : int, number of basis functions
        - kernel : dict, kernel specification (e.g., {'type': 'gaussian', 'std': 1.0})
        - grid_range : tuple, basis function placement range
        
        **Sine layer:**
        - D : int, number of basis functions
    seed : int, default=42
        Random seed for reproducible weight initialization.

    Attributes
    ----------
    layer_type : str
        The type of layers used in the network (lowercase).
    layers : List[nnx.Module]
        List of KAN layer instances that compose the network.

    Methods
    -------
    update_grids(x, G_new)
        Update grid resolution for all layers based on input data.
    __call__(x)
        Forward pass through the network.

    Examples
    --------
    **Basic spline-based KAN:**
    
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxkan import KAN
    >>> 
    >>> # Create a 2-layer spline KAN
    >>> spline_params = {'k': 3, 'G': 5}
    >>> model = KAN(layer_dims=[2, 5, 1], layer_type='base', 
    ...             required_parameters=spline_params, seed=42)
    >>> 
    >>> # Generate sample data
    >>> key = jax.random.key(42)
    >>> x = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
    >>> 
    >>> # Forward pass
    >>> output = model(x)
    >>> print(output.shape)  # (100, 1)
    
    **Chebyshev polynomial KAN:**
    
    >>> cheby_params = {'D': 5, 'flavor': 'default'}
    >>> model = KAN([3, 8, 8, 1], 'chebyshev', cheby_params, seed=42)
    >>> x = jax.random.normal(jax.random.key(0), (50, 3))
    >>> output = model(x)
    
    **Fourier-based KAN:**
    
    >>> fourier_params = {'D': 10, 'smooth_init': True}
    >>> model = KAN([2, 6, 1], 'fourier', fourier_params, seed=42)
    >>> x = jax.random.uniform(jax.random.key(1), (32, 2), minval=-2, maxval=2)
    >>> output = model(x)
    
    **RBF KAN with Gaussian kernels:**
    
    >>> rbf_params = {
    ...     'D': 8, 
    ...     'kernel': {'type': 'gaussian', 'std': 1.0},
    ...     'grid_range': (-2.0, 2.0)
    ... }
    >>> model = KAN([4, 10, 1], 'rbf', rbf_params, seed=42)
    >>> x = jax.random.normal(jax.random.key(2), (64, 4))
    >>> output = model(x)
    
    **Adaptive grid refinement:**
    
    >>> # Start with coarse grid
    >>> params = {'k': 3, 'G': 3}
    >>> model = KAN([2, 5, 1], 'base', params, seed=42)
    >>> 
    >>> # Generate training data
    >>> x_train = jax.random.uniform(jax.random.key(0), (1000, 2), minval=-1, maxval=1)
    >>> 
    >>> # Refine grid based on data distribution
    >>> model.update_grids(x_train, G_new=10)

    Notes
    -----
    - **Grid Adaptation**: Spline-based layers (base, spline) support adaptive grid 
      refinement through the `update_grids` method, which repositions knots based on 
      input data distribution for better approximation.
      
    - **Polynomial Layers**: Chebyshev and Legendre layers use `update_grid` to increase
      polynomial degree rather than adjusting spatial grids.
      
    - **Memory Considerations**: Higher grid resolutions (G) or polynomial degrees (D) 
      increase memory usage and computation time. Start with moderate values and 
      increase as needed.
      
    - **Numerical Stability**: Some layer types (especially high-degree polynomials) 
      may suffer from numerical instability. Consider using appropriate initialization
      schemes and regularization.
      
    - **Activation Functions**: Base and spline layers support residual connections 
      with traditional activation functions (e.g., SiLU) for improved expressivity.

    References
    ----------
    .. [1] Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks." 
           arXiv preprint arXiv:2404.19756 (2024).
    .. [2] Kolmogorov, A. N. "On the representation of continuous functions of many 
           variables by superposition of continuous functions of one variable and 
           addition." Proceedings of the USSR Academy of Sciences 114 (1957): 953-956.
    .. [3] Arnold, V. I. "On the representation of functions of several variables as a 
           superposition of functions of a smaller number of variables." 
           Mathematical Problems of Cybernetics 2 (1961): 41-61.

    See Also
    --------
    jaxkan.layers : Individual KAN layer implementations
    jaxkan.utils.PIKAN : Physics-Informed KAN utilities for solving PDEs
    """
    
    def __init__(self, 
                 layer_dims: List[int], 
                 layer_type: Literal['base', 'spline', 'chebyshev', 'legendre', 'fourier', 'rbf', 'sine'] = "base",
                 required_parameters: Union[dict, None] = None, 
                 seed: int = 42
                ) -> None:
        """
        Initializes a KAN model.

        Parameters
        ----------
        layer_dims : List[int]
            Defines the network architecture in terms of nodes per layer.
            For example, [4,5,1] creates a network with 2 layers:
            - First layer: 4 inputs -> 5 outputs
            - Second layer: 5 inputs -> 1 output
        layer_type : {'base', 'spline', 'chebyshev', 'legendre', 'fourier', 'rbf', 'sine'}
            Type of layer to use throughout the network. See class docstring for details
            on each layer type's characteristics and requirements.
        required_parameters : dict
            Dictionary containing parameters required for the chosen layer type.
            Required parameters vary by layer type - see class docstring for the
            full specification.
        seed : int, default=42
            Random seed for reproducible weight initialization.

        Examples
        --------
        Create a spline-based KAN with 3rd order splines and 5 grid intervals:

        >>> req_params = {'k': 3, 'G': 5}  
        >>> model = KAN(layer_dims=[2,5,1], layer_type='base', 
        ...            required_parameters=req_params, seed=42)

        Notes
        -----
        The required_parameters dict must match the requirements of the chosen
        layer_type. Failing to provide the correct parameters will raise a
        ValueError.
        """
        self.layer_type = layer_type.lower()
        
        # Get the corresponding layer class based on layer_type
        LayerClass = get_layer(self.layer_type)
            
        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")
        
        self.layers = [
                LayerClass(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    seed=seed
                )
                for i in range(len(layer_dims) - 1)
            ]
    
    def update_grids(self, x: jax.Array, G_new: int) -> None:
        """
        Update grid resolution for each layer of the KAN architecture.

        This method adaptively refines the grid/basis functions based on the input data 
        distribution. For spline-based layers, it repositions knot points to better 
        capture the data characteristics. For polynomial layers, it increases the 
        polynomial degree. For other layer types, it adjusts the number of basis functions.

        Parameters
        ----------
        x : jax.Array
            Input data for grid adaptation with shape (batch_size, n_features).
            Should represent the data distribution that the network will encounter
            during training or inference.
        G_new : int
            New grid resolution parameter. Interpretation depends on layer type:
            
            - For base/spline layers: Number of grid intervals
            - For polynomial layers: New polynomial degree  
            - For Fourier/RBF/Sine layers: Number of basis functions

        Raises
        ------
        ValueError
            If G_new is not a positive integer or if the layer type doesn't support
            grid updates.
        RuntimeError
            If the grid update fails due to numerical issues or incompatible data.

        Notes
        -----
        - **Progressive Refinement**: It's recommended to gradually increase G_new 
          rather than making large jumps to maintain numerical stability.
          
        - **Data Dependency**: The quality of grid adaptation depends on how well 
          the input data `x` represents the true data distribution.
          
        - **Memory Usage**: Higher G_new values increase memory consumption and 
          computational cost. Monitor resource usage when increasing grid resolution.
          
        - **Layer Propagation**: The method processes layers sequentially, using 
          the output of each layer as input to the next layer's grid update.

        Examples
        --------
        **Basic grid refinement:**
        
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from jaxkan import KAN
        >>> 
        >>> # Create model with coarse initial grid
        >>> params = {'k': 3, 'G': 3}
        >>> model = KAN([2, 5, 1], 'base', params, seed=42)
        >>> 
        >>> # Generate representative training data
        >>> key = jax.random.key(42)
        >>> x_train = jax.random.uniform(key, (1000, 2), minval=-1.0, maxval=1.0)
        >>> 
        >>> # Refine grid based on data
        >>> model.update_grids(x_train, G_new=10)
        
        **Progressive refinement for polynomial KAN:**
        
        >>> cheby_params = {'D': 3, 'flavor': 'default'}
        >>> model = KAN([3, 8, 1], 'chebyshev', cheby_params, seed=42)
        >>> x = jax.random.normal(jax.random.key(0), (500, 3))
        >>> 
        >>> # Gradually increase polynomial degree
        >>> model.update_grids(x, D_new=5)
        >>> model.update_grids(x, D_new=8)
        
        **Adaptive training workflow:**
        
        >>> # Training loop with periodic grid refinement
        >>> for epoch in range(1000):
        ...     # ... training code ...
        ...     if epoch % 200 == 0 and epoch > 0:
        ...         model.update_grids(x_batch, G_new=min(15, 5 + epoch//200))

        See Also
        --------
        jaxkan.layers.BaseLayer.update_grid : Individual layer grid update
        jaxkan.utils.PIKAN.train_PIKAN : Automated grid adaptation during training
        """

        # Loop over each layer
        for i, layer in enumerate(self.layers):
            
            # Update the grid for the current layer
            layer.update_grid(x, G_new)

            # Perform a forward pass to get the input for the next layer
            x = layer(x)

    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the KAN network.

        Sequentially applies each KAN layer to transform the input through learned
        univariate functions. Each layer applies basis functions (splines, polynomials,
        Fourier modes, etc.) along network edges and combines them to produce the
        layer output.

        Parameters
        ----------
        x : jax.Array
            Input tensor with shape (batch_size, n_input_features).
            The number of input features must match the first dimension 
            in `layer_dims` specified during initialization.

        Returns
        -------
        jax.Array
            Network output with shape (batch_size, n_output_features).
            The number of output features matches the last dimension 
            in `layer_dims`.

        Raises
        ------
        ValueError
            If input shape doesn't match the expected network input dimension.
        RuntimeError
            If forward pass fails due to numerical issues or uninitialized layers.

        Notes
        -----
        - **Automatic Differentiation**: This method is fully compatible with JAX's
          automatic differentiation for gradient computation.
          
        - **JIT Compilation**: Can be wrapped with `jax.jit` for improved performance
          on repeated calls with similar input shapes.
          
        - **Vectorization**: Supports batched inputs for efficient parallel processing.
          
        - **Memory Efficiency**: For very large batch sizes, consider processing 
          in smaller chunks to manage memory usage.

        Examples
        --------
        **Basic inference:**
        
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from jaxkan import KAN
        >>> 
        >>> # Create and initialize model
        >>> params = {'k': 3, 'G': 5}
        >>> model = KAN([2, 8, 1], 'base', params, seed=42)
        >>> 
        >>> # Single sample inference
        >>> x_single = jnp.array([[0.5, -0.3]])
        >>> output = model(x_single)
        >>> print(output.shape)  # (1, 1)
        
        **Batch processing:**
        
        >>> # Multiple samples
        >>> key = jax.random.key(42)
        >>> x_batch = jax.random.uniform(key, (100, 2), minval=-1, maxval=1)
        >>> outputs = model(x_batch)
        >>> print(outputs.shape)  # (100, 1)
        
        **JIT compilation for speed:**
        
        >>> # Compile for faster repeated calls
        >>> compiled_model = jax.jit(model)
        >>> fast_outputs = compiled_model(x_batch)
        
        **Gradient computation:**
        
        >>> # Define loss function
        >>> def loss_fn(params, x, y_true):
        ...     y_pred = model(x)
        ...     return jnp.mean((y_pred - y_true)**2)
        >>> 
        >>> # Compute gradients
        >>> grad_fn = jax.grad(loss_fn)
        >>> grads = grad_fn(model, x_batch, y_true)
        
        **Function approximation example:**
        
        >>> # Approximate f(x1, x2) = x1*sin(x2) + x2*cos(x1)
        >>> def target_fn(x):
        ...     return x[:, 0:1] * jnp.sin(x[:, 1:2]) + x[:, 1:2] * jnp.cos(x[:, 0:1])
        >>> 
        >>> # Generate training data
        >>> x_train = jax.random.uniform(jax.random.key(0), (1000, 2), 
        ...                              minval=-jnp.pi, maxval=jnp.pi)
        >>> y_train = target_fn(x_train)
        >>> 
        >>> # Test model approximation
        >>> y_pred = model(x_train)
        >>> error = jnp.mean(jnp.abs(y_pred - y_train))

        See Also
        --------
        jax.jit : Just-in-time compilation for performance
        jax.grad : Gradient computation
        jax.vmap : Vectorized mapping over batch dimension
        """

        # Pass through each layer of the KAN
        for _, layer in enumerate(self.layers):
            x = layer(x)

        return x
