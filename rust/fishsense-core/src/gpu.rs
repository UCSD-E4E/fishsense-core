use tracing::info;
use wgpu::{Device, PowerPreference, Queue};

use crate::errors::FishSenseError;

pub async fn get_device_and_queue() -> Result<(Device, Queue), FishSenseError> {
    let instance = wgpu::Instance::default();

    let request_options = wgpu::RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        ..Default::default()
    };

    let adapter = instance
        .request_adapter(&request_options)
        .await;

    let adapter = adapter.map_err(|_| FishSenseError::CannotAcquireGpu)?;

    info!(adapter = %adapter.get_info().name, "acquired GPU adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            },
        )
        .await?;

    Ok((device, queue))
}