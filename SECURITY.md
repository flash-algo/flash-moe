# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Security Considerations

flash-moe is a high-performance Mixture of Experts implementation, primarily achieved through two modules: a Cartesian Product Router and Sparse Feedforward Networks.

When using this repository:

- Only run code you understand, especially when modifying kernels.
- Use virtual environments to isolate dependencies.
- Be careful when experimenting with very large tensor sizes, as they may cause out-of-memory errors.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**For security issues:**
- Email: losercheems@gmail.com
- Subject: [SECURITY] flash-moe Vulnerability Report
- Include: Detailed description, reproduction steps, and potential impact

**For general bugs:**
- Use our [GitHub Issues](https://github.com/flash-algo/flash-moe/issues)
- Follow our [contributing guidelines](CONTRIBUTING.md)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity and complexity

Critical security issues will be prioritized and may result in emergency releases.

## Security Best Practices

When using flash-moe:

1. **Environment Isolation**
   ```bash
   # Use virtual environments
   python -m venv fmoe_env
   source fmoe_env/bin/activate  # Linux/Mac
   # or
   fmoe_env\Scripts\activate     # Windows
   ```

2. **Dependency Management**
   ```bash
   # Keep dependencies updated
   pip install --upgrade torch flash-moe
   ```

3. **Input Validation**
   ```python
   # Validate tensor shapes and dtypes before processing
   assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
   assert x.shape == y.shape
   ```

4. **Resource Monitoring**
   ```python
   # Monitor GPU memory usage
   import torch
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

## Disclosure Policy

- Confirmed vulnerabilities will be disclosed responsibly
- Security fixes will be released as soon as safely possible
- CVE numbers will be requested for significant vulnerabilities
- Credit will be given to security researchers who report issues responsibly

## Contact

For security-related questions or concerns:
- Primary: losercheems@gmail.com
- Project maintainers: See [GitHub contributors](https://github.com/flash-algo/flash-moe/graphs/contributors)

For general support:
- GitHub Issues: https://github.com/flash-algo/flash-moe/issues
- Documentation: see the main README and docs/ in this repository.