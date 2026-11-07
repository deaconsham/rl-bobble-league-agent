/*
The ball. Caps its height so it can never clear a wall, and brakes smoothly to a stop.
*/

using Godot;
using System;

public partial class Ball : RigidBody3D {
	// set by the game manager so training and human play match
	float brakeSpeed = 1.0f;
	float brakeDecel = 3.0f;

	// under this side to side speed it counts as stopped (too small to see the snap)
	const float RestSpeed = 0.05f;

	public void SetBrake(float speed, float decel) {
		brakeSpeed = speed;
		brakeDecel = decel;
	}

	// Highest the ball's centre may go. Arena walls are 1.39 tall (2.7854 * 0.5 scale)
	// and the ball radius is 0.5, so the ball clears a wall only above y = 1.89.
	// 1.2 still allows a small hop but the ball can never get over a wall.
	[Export] float maxHeight = 1.2f;

	// godot's safe place to change a body's velocity
	public override void _IntegrateForces(PhysicsDirectBodyState3D state) {
		Transform3D t = state.Transform;
		if (t.Origin.Y > maxHeight) {
			t.Origin = new Vector3(t.Origin.X, maxHeight, t.Origin.Z);
			state.Transform = t;
			Vector3 v = state.LinearVelocity;
			if (v.Y > 0) state.LinearVelocity = new Vector3(v.X, 0, v.Z);
		}
		Brake(state, brakeSpeed, brakeDecel, RestSpeed);
	}

	// shared with Player. under brakeSpeed, lose decel * dt of speed each tick.
	// spin is scaled the same so it can't roll the ball forward again
	public static void Brake(PhysicsDirectBodyState3D state, float brakeSpeed, float decel, float restSpeed) {
		Vector3 v = state.LinearVelocity;
		float speed = new Vector2(v.X, v.Z).Length();
		if (speed >= brakeSpeed) return;
		float slowed = speed - decel * state.Step;
		if (slowed <= restSpeed) {
			state.LinearVelocity = new Vector3(0, v.Y, 0);
			state.AngularVelocity = Vector3.Zero;
			return;
		}
		float k = slowed / speed;
		state.LinearVelocity = new Vector3(v.X * k, v.Y, v.Z * k);
		state.AngularVelocity *= k;
	}
}
